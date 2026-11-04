"""Builds verl parquet files. math: DAPO-Math-17k (train) + AIME24/25, HMMT-Feb-25, AMC23, MATH-500 (eval).
sdpo: the SciKnowEval {biology,chemistry,material,physics} and ToolUse train/test splits shipped with github.com/lasgroup/SDPO."""

import argparse
import json
import os
import re

import pandas as pd
from datasets import load_dataset

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DAPO_HEAD = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
DAPO_TAIL = '\n\nRemember to put your answer on its own line after "Answer:".'
SDPO_TASKS = {t: f"sciknoweval/{t}" for t in ("biology", "chemistry", "material", "physics")} | {"tooluse": "tooluse"}


def row(source, problem, answer, idx, split):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem.strip() + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {"index": idx, "split": split, "answer": str(answer).strip()},
    }


def dapo_rows():
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    seen, out = set(), []
    for ex in ds:
        text = ex["prompt"][0]["content"]
        if text in seen:
            continue
        seen.add(text)
        problem = text.removeprefix(DAPO_HEAD).removesuffix(DAPO_TAIL)
        out.append(row("dapo_math", problem, ex["reward_model"]["ground_truth"], len(out), "train"))
    return out


def eval_rows():
    specs = {
        "aime24": ("math-ai/aime24", "test", "problem", lambda ex: re.search(r"\\boxed\{(.*)\}", ex["solution"]).group(1)),
        "aime25": ("math-ai/aime25", "test", "problem", lambda ex: ex["answer"]),
        "hmmt_feb25": ("MathArena/hmmt_feb_2025", "train", "problem", lambda ex: ex["answer"]),
        "amc23": ("math-ai/amc23", "test", "question", lambda ex: ex["answer"]),
        "math500": ("HuggingFaceH4/MATH-500", "test", "problem", lambda ex: ex["answer"]),
    }
    out = {}
    for name, (repo, split, key, answer) in specs.items():
        ds = load_dataset(repo, split=split)
        out[name] = [row(name, ex[key], answer(ex), i, "test") for i, ex in enumerate(ds)]
    return out


def sdpo_rows(task, path, split):
    out = []
    for line in open(path):
        ex = json.loads(line)
        system = [{"role": "system", "content": ex["system"]}] if ex.get("system") else []
        out.append({
            "data_source": task,
            "prompt": system + [{"role": "user", "content": ex["prompt"]}],
            "ability": ex["kind"],
            "reward_model": {"style": ex["kind"], "ground_truth": ex["answer"]},
            "extra_info": {"index": ex["idx"], "split": split},
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/dev/shm/looprl/data")
    p.add_argument("--sets", nargs="+", default=["math", "sdpo"])
    p.add_argument("--sdpo-dir", default="/workspace/SDPO/datasets")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if "math" in args.sets:
        train = dapo_rows()
        pd.DataFrame(train).to_parquet(f"{args.out}/dapo_math_train.parquet")
        print("train", len(train))
        evals = eval_rows()
        for name, rows in evals.items():
            pd.DataFrame(rows).to_parquet(f"{args.out}/{name}.parquet")
            print(name, len(rows))
        val = evals["aime24"] + evals["aime25"] + evals["amc23"]
        pd.DataFrame(val).to_parquet(f"{args.out}/val_competition.parquet")
        print("val_competition", len(val))

    if "sdpo" in args.sets:
        for task, sub in SDPO_TASKS.items():
            os.makedirs(f"{args.out}/sdpo/{task}", exist_ok=True)
            for split in ("train", "test"):
                rows = sdpo_rows(task, f"{args.sdpo_dir}/{sub}/{split}.json", split)
                pd.DataFrame(rows).to_parquet(f"{args.out}/sdpo/{task}/{split}.parquet")
                print(f"sdpo/{task}/{split}", len(rows))


if __name__ == "__main__":
    main()
