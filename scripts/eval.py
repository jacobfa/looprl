"""Evaluates a Dory checkpoint with vLLM at several loop counts (avg@n / pass@n) and logs to wandb."""

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import wandb
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import looprl
from looprl.reward import compute_score

PKG = os.path.dirname(looprl.__file__)
BASE = "/dev/shm/looprl/models/dory-4b-hf"


def ensure_loadable(model_dir):
    for f in ("configuration_dory.py", "modeling_dory.py"):
        if not os.path.exists(f"{model_dir}/{f}"):
            shutil.copy(f"{PKG}/{f}", model_dir)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if not os.path.exists(f"{model_dir}/{f}") and os.path.exists(f"{BASE}/{f}"):
            shutil.copy(f"{BASE}/{f}", model_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--datasets", nargs="+", default=["aime24", "aime25", "hmmt_feb25", "amc23", "math500"],
                   help="parquet names under --data, e.g. aime24 or sdpo/chemistry/test")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--n-large", type=int, default=4, help="samples per prompt for datasets with > 100 problems")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-thinking", action="store_true", help="enable_thinking=False in the chat template (SDPO setup)")
    p.add_argument("--data", default="/dev/shm/looprl/data")
    p.add_argument("--out", default="/dev/shm/looprl/eval")
    p.add_argument("--project", default="looprl")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    args = p.parse_args()

    ensure_loadable(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    out_dir = f"{args.out}/{args.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    run = wandb.init(project=args.project, name=f"eval-{args.run_name}", dir="/dev/shm/looprl/wandb", config=vars(args))
    table = wandb.Table(columns=["loops", "dataset", "n", "avg@n", "pass@n", "has_answer", "resp_len", "truncated"])

    for loops in args.loops:
        llm = LLM(
            model=args.model, trust_remote_code=True, tensor_parallel_size=1, dtype="bfloat16",
            hf_overrides={"n_recurrent_loops": loops}, max_model_len=2048 + args.max_tokens,
            gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, seed=0,
        )
        for ds in args.datasets:
            df = pd.read_parquet(f"{args.data}/{ds}.parquet")
            n = args.n if len(df) <= 100 else args.n_large
            kw = {"enable_thinking": False} if args.no_thinking else {}
            prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True, **kw) for m in df["prompt"]]
            sp = SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=0)
            t0 = time.time()
            outs = llm.generate(prompts, sp)
            acc, lens, trunc, has = [], [], [], []
            with open(f"{out_dir}/{ds.replace('/', '_')}_loops{loops}.jsonl", "w") as f:
                for row, o in zip(df.itertuples(), outs):
                    gt = row.reward_model["ground_truth"]
                    scores = [compute_score(row.data_source, c.text, gt) for c in o.outputs]
                    acc.append([s["acc"] for s in scores])
                    has += [s["has_answer"] for s in scores]
                    lens += [len(c.token_ids) for c in o.outputs]
                    trunc += [c.finish_reason == "length" for c in o.outputs]
                    f.write(json.dumps({"idx": row.Index, "gt": gt, "acc": acc[-1], "texts": [c.text for c in o.outputs]}) + "\n")
            acc = np.array(acc)
            m = {
                "avg@n": acc.mean(), "pass@n": acc.max(1).mean(), "has_answer": np.mean(has),
                "resp_len": np.mean(lens), "truncated": np.mean(trunc), "n": n, "secs": time.time() - t0,
            }
            print(f"[eval] loops={loops} {ds}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            run.log({f"{ds}/loops{loops}/{k}": v for k, v in m.items()})
            table.add_data(loops, ds, n, m["avg@n"], m["pass@n"], m["has_answer"], m["resp_len"], m["truncated"])
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    run.log({"results": table})
    run.finish()


if __name__ == "__main__":
    main()
