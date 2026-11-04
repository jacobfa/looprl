"""Binary verifiable rewards: math (last \\boxed{} via math-verify), SciKnowEval MCQ (<answer>X</answer>), ToolUse (Action + JSON input)."""

import json
import re
from collections import Counter

from math_verify import parse, verify

BOXED = re.compile(r"\\boxed\s*\{")
MCQ_FORMAT = re.compile(r"<answer>\s*[ABCD]\s*</answer>\s*$")
MCQ_SOURCES = {"biology", "chemistry", "material", "physics"}


def last_boxed(text):
    m = None
    for m in BOXED.finditer(text):
        pass
    if m is None:
        return None
    i, depth = m.end(), 1
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end() : i - 1].strip() if depth == 0 else None


def normalize(s):
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    return s.replace(" ", "").replace("$", "").rstrip(".").strip()


def is_equiv(pred, gt):
    if pred is None:
        return False
    if normalize(pred) == normalize(gt):
        return True
    try:
        return verify(parse(f"${gt}$"), parse(f"${pred}$"))
    except Exception:
        return False


def math_score(text, gt):
    pred = last_boxed(text)
    return float(is_equiv(pred, gt)), float(pred is not None)


def mcq_score(text, gt):
    pred = text.split("<answer>")[-1].split("</answer>")[0].strip()
    return float(pred == gt), float(MCQ_FORMAT.search(text) is not None)


def _loads(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except json.JSONDecodeError:
        return None


def tooluse_score(text, gt):
    items = _loads(gt) or []
    gt_actions = Counter(x["Action"] for x in items)
    gt_inputs = {k: v for x in items for k, v in (_loads(x.get("Action_Input")) or {}).items()}
    actions = Counter(re.findall(r"Action:\s*(\w+)", text))
    inputs = {k: v for b in re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL) for k, v in (_loads(b) or {}).items()}
    acc = float(actions == gt_actions and inputs == gt_inputs)
    return acc, float(re.search(r"Action:.*?\nAction Input:", text, re.DOTALL) is not None)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = str(ground_truth)
    if data_source == "tooluse":
        acc, has = tooluse_score(solution_str, gt)
    elif data_source in MCQ_SOURCES:
        acc, has = mcq_score(solution_str, gt)
    else:
        acc, has = math_score(solution_str, gt)
    return {"score": acc, "acc": acc, "has_answer": has}
