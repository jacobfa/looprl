"""Strictly on-policy policy-gradient losses for looped LMs: GRPO, RLTT and loop-residual PG (LRPG)."""

import torch

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import no_padding_2_padding

from .engine import AUX_KEYS, patch

patch()


def loop_weights(n_loops, scheme, device):
    t = torch.arange(1, n_loops + 1, device=device, dtype=torch.float32)
    w = torch.ones_like(t) if scheme == "uniform" else t if scheme == "progressive" else None
    if w is None:
        raise ValueError(scheme)
    return w / w.sum()


def masked_corr(a, b, mask):
    a = a - masked_mean(a, mask)
    b = b - masked_mean(b, mask)
    return masked_mean(a * b, mask) / (masked_mean(a * a, mask).sqrt() * masked_mean(b * b, mask).sqrt() + 1e-8)


def loop_pg_loss(config, model_output, data, dp_group=None, method="grpo", alpha=1.0, weights="uniform", lam=0.0):
    """method: grpo | rltt | lrpg; alpha: LRPG residual coefficient; weights: RLTT loop weights;
    lam: deep-supervision weight, adds lam * sg(A) * mean_l log pi^(l)(y) over the non-final loops."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    aux = {k: no_padding_2_padding(model_output[k], data) for k in AUX_KEYS if k in model_output}

    gbi = config.global_batch_info
    gbi.update(
        dp_size=data["dp_size"],
        batch_num_tokens=data["batch_num_tokens"],
        global_batch_size=data["global_batch_size"],
        loss_scale_factor=config.loss_scale_factor,
    )
    is_global = data["dp_size"] > 1 or data["batch_num_tokens"] is not None or data["global_batch_size"] is not None
    agg_type = AggregationType.SUM if is_global or config.loss_scale_factor is not None else AggregationType.MEAN

    d = data.select("response_mask", "old_log_probs", "advantages").to_padded_tensor()
    mask = d["response_mask"].to(bool)
    adv = adv0 = d["advantages"].detach()
    metrics, sums = {}, {} 

    if "loop_logp" in aux:
        metrics["loop/logp_gap"] = masked_mean(log_prob.detach() - aux["loop_logp"][..., -1].detach(), mask)
        for i in range(aux["loop_logp"].shape[-1]):
            metrics[f"loop/logp_l{i + 1}"] = masked_mean(aux["loop_logp"][..., i].detach(), mask)

    if method == "rltt":
        logps = torch.cat([aux["loop_logp"], log_prob.unsqueeze(-1)], dim=-1)
        score = (logps * loop_weights(logps.shape[-1], weights, logps.device)).sum(-1)
    elif method == "lrpg":
        d_y, div_pq = aux["d_y"].detach(), aux["div_pq"]
        metrics.update(
            {
                "loop/d_y_mean": masked_mean(d_y, mask),
                "loop/d_y_std": masked_mean(d_y.pow(2), mask).sqrt(),
                "loop/div_pq": masked_mean(div_pq.detach(), mask),
                "loop/logp_gap": masked_mean(log_prob.detach() - aux["logp_prev"], mask),
                "loop/corr_adv_d_y": masked_corr(adv, d_y, mask),
            }
        )
        adv = adv - alpha * d_y
        score = log_prob
        # batch sums of <g_t, c_t> and |c_t|^2 (fitting the residual coefficient); the trainer refits alpha
        uu, upd, pdd = aux["alpha_stats"].unbind(-1)
        metrics["loop/alpha"] = alpha
        sums = {
            "loop/alpha_num": (adv0 * (d_y * uu - upd) * mask).sum(),
            "loop/alpha_den": ((d_y.pow(2) * uu - 2 * d_y * upd + pdd) * mask).sum(),
        }
    else:
        score = log_prob

    loss_mat = -adv * torch.exp(score - score.detach())
    if method == "lrpg":
        loss_mat = loss_mat - alpha * aux["div_pq"]
    if lam > 0:
        lp = aux["loop_logp"]
        loss_mat = loss_mat - lam * adv0 * torch.exp(lp - lp.detach()).mean(-1)
    pg_loss = agg_loss(loss_mat=loss_mat, loss_mask=mask, loss_agg_mode=config.loss_agg_mode, **gbi)
    metrics["loop/logp"] = masked_mean(log_prob.detach(), mask)
    metrics["actor/ppo_kl"] = masked_mean(d["old_log_probs"] - log_prob.detach(), mask)
    metrics = Metric.from_dict(metrics, aggregation=AggregationType.MEAN)
    metrics.update(Metric.from_dict(sums, aggregation=AggregationType.SUM))
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=agg_type)

    loss = pg_loss
    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
        ent_loss = agg_loss(loss_mat=entropy, loss_mask=mask, loss_agg_mode=config.loss_agg_mode, **gbi)
        loss = loss - config.entropy_coeff * ent_loss
        metrics["actor/entropy_loss"] = Metric(value=ent_loss, aggregation=agg_type)
    return loss, metrics
