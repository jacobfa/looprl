"""verl PPO entry point with the loop-aware policy loss. Config knobs live under `looprl.*` (see scripts/train.sh)."""

import re
import time
from datetime import datetime, timedelta
from functools import partial

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device
from verl.workers.config import ActorConfig

VAL_KEY = re.compile(r"val-(?:core|aux)/([^/]+)/acc/(?:mean@\d+|best@(\d+)/mean)")
SUMMARY_KEYS = (
    "critic/score/mean", "response_length/mean", "response_length/clip_ratio", "actor/pg_loss", "actor/entropy",
    "actor/grad_norm", "actor/ppo_kl", "actor/lr", "timing_s/gen", "timing_s/old_log_prob", "timing_s/update_actor",
    "timing_s/testing", "timing_s/step", "perf/throughput",
)


class LoopTrainer(PPOTrainerSync):
    def _setup(self):
        super()._setup()
        from looprl.losses import loop_pg_loss

        actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        knobs = OmegaConf.to_container(self.config.looprl, resolve=True)
        self.max_train_hours, self.fit_alpha, self.alpha = (knobs.pop(k) for k in ("max_train_hours", "fit_alpha", "alpha"))
        self._loss_fn = partial(loop_pg_loss, config=actor_config, **knobs)
        self.actor_rollout_wg.set_loss_fn(partial(self._loss_fn, alpha=self.alpha))
        self._t0, self._step0, self._train_s, self._best = time.time(), self.global_steps, 0.0, {}
        self._val_n = str(self.config.actor_rollout_ref.rollout.val_kwargs.n)

    def _validate(self):
        metrics = super()._validate()
        for k in [k for k in metrics if k.startswith("val-core/") and "/acc/mean@" in k]:
            self._best[k] = metrics[k.replace("val-core/", "val-best/")] = max(self._best.get(k, 0.0), metrics[k])
        return metrics

    def _compute_metrics(self, batch, metrics, timing_raw, **kwargs):
        super()._compute_metrics(batch, metrics, timing_raw, **kwargs)
        if self.fit_alpha and "actor/loop/alpha_den" in metrics:  # alpha_next = sum<g,c> / (sum|c|^2 + eps), fixed for next batch
            self.alpha = float(metrics["actor/loop/alpha_num"] / (metrics["actor/loop/alpha_den"] + 1e-6))
            metrics["actor/loop/alpha_next"] = self.alpha
            self.actor_rollout_wg.set_loss_fn(partial(self._loss_fn, alpha=self.alpha))
        self._train_s += timing_raw["step"] - timing_raw.get("save_checkpoint", 0.0)
        done = self.global_steps - self._step0 + 1
        wall = time.time() - self._t0
        if self.max_train_hours:
            budget = self.max_train_hours * 3600
            if self._train_s * (1 + 1 / done) >= budget:  # next step would exhaust the budget -> make it the last one
                self.total_training_steps = min(self.total_training_steps, self.global_steps + 1)
            eta = max(budget - self._train_s, 0.0) * wall / self._train_s
            head = f"[step {self.global_steps} | {self._train_s / 3600:.2f}/{self.max_train_hours:g} train h]"
        else:
            eta = wall / done * (self.total_training_steps - self.global_steps)
            head = f"[step {self.global_steps}/{self.total_training_steps}]"
        metrics["progress/train_hours"] = self._train_s / 3600
        metrics["progress/min_per_step"] = wall / done / 60
        metrics["progress/eta_hours"] = eta / 3600
        shown = {k.removeprefix("actor/") if "loop/" in k else k: v for k, v in metrics.items() if k in SUMMARY_KEYS or "loop/" in k}
        val = {}  # val-*/<dataset>/acc/{mean@n,best@k/mean} -> <dataset>@{1,4,n}
        for k, v in metrics.items():
            if (m := VAL_KEY.fullmatch(k)) and m[2] in (None, "4", self._val_n):
                val[m[1], int(m[2] or 1)] = v
        shown |= {f"{ds}@{k}": val[ds, k] for ds, k in sorted(val)}
        print(
            f"{head} " + " ".join(f"{k.split('/', 1)[1] if k.startswith('timing_s') else k}={v:.4g}" for k, v in shown.items())
            + f" | {wall / done / 60:.1f} min/step, elapsed {timedelta(seconds=int(wall))}, "
            f"remaining {timedelta(seconds=int(eta))}, finish ~{datetime.now() + timedelta(seconds=eta):%m-%d %H:%M}",
            flush=True,
        )


@ray.remote
class LoopTaskRunner(TaskRunnerV1.__ray_actor_class__):
    def run(self, config):
        trainer_base.TRAINER_REGISTRY["sync"] = LoopTrainer
        super().run(config)


@hydra.main(config_path="../config", config_name="looprl", version_base=None)
def main(config):
    model_cfg = config.actor_rollout_ref.model.override_config
    model_cfg.rl_aux = None if config.looprl.method == "grpo" else config.looprl.method
    model_cfg.rl_deep_supervision = config.looprl.lam > 0
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=LoopTaskRunner)


if __name__ == "__main__":
    main()
