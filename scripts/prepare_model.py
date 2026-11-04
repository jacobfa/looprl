"""Download Dory and build a local checkpoint dir that loads via trust_remote_code (HF) and the vLLM plugin."""

import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "looprl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
    ap.add_argument("--root", default="/dev/shm/looprl/models")
    ap.add_argument("--name", default="dory-4b")
    ap.add_argument("--loops", type=int, default=2)
    args = ap.parse_args()

    src = snapshot_download(args.repo, local_dir=os.path.join(args.root, f"{args.name}-hf"),
                            allow_patterns=["*.json", "*.safetensors", "*.jinja"])
    dst = os.path.join(args.root, f"{args.name}-L{args.loops}")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f == "config.json" or f.startswith("."):
            continue
        link = os.path.join(dst, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, f), link)
    for f in ("modeling_dory.py", "configuration_dory.py"):
        shutil.copy(os.path.join(PKG, f), os.path.join(dst, f))

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["block_types"] = cfg.pop("layer_types")
    cfg["n_recurrent_loops"] = args.loops
    cfg["recurrent_kv_cache_mode"] = "per_loop"
    cfg["auto_map"] = {
        "AutoConfig": "configuration_dory.DoryConfig",
        "AutoModelForCausalLM": "modeling_dory.DoryForCausalLM",
    }
    cfg["use_cache"] = False
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(dst)


if __name__ == "__main__":
    main()
