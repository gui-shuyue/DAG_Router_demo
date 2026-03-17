import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def ensure_question_prefix(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return "Question: "
    if s.lower().startswith("question:"):
        return s
    return f"Question: {s}"


def to_string(v: Any) -> str:
    return "" if v is None else str(v)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dag_math(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        pid = to_string(it.get("problem_id", i))
        out.append(
            {
                "problem_id": f"dag_math_mini:{pid}",
                "subject": to_string(it.get("subject", "math")),
                "problem": ensure_question_prefix(it.get("problem", "")),
                "answer": to_string(it.get("answer", "")).strip(),
            }
        )
    return out


def normalize_prism(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        pid = to_string(it.get("problem_id", i))
        out.append(
            {
                "problem_id": f"prism_physics:{pid}",
                "subject": to_string(it.get("subject", "physics")),
                "problem": ensure_question_prefix(it.get("problem", "")),
                "answer": to_string(it.get("answer", "")).strip(),
            }
        )
    return out


def normalize_hotpot(payload: Any) -> List[Dict[str, str]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = []
        for value in payload.values():
            if isinstance(value, list):
                items.extend(value)
    else:
        items = []

    out: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        pid = to_string(it.get("unique_id", i))
        out.append(
            {
                "problem_id": f"hotpotqa:{pid}",
                "subject": to_string(it.get("subject", "qa")),
                "problem": ensure_question_prefix(it.get("problem", "")),
                "answer": to_string(it.get("answer", "")).strip(),
            }
        )
    return out


def normalize_mmlu(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        pid = to_string(it.get("id", i))
        out.append(
            {
                "problem_id": f"mmlu_pro_mini:{pid}",
                "subject": to_string(it.get("category", "mmlu")),
                "problem": ensure_question_prefix(it.get("problem_text", "")),
                "answer": to_string(it.get("ground_truth_answer", "")).strip(),
            }
        )
    return out


def proportional_counts(counts: Dict[str, int], total_target: int) -> Dict[str, int]:
    total_available = sum(counts.values())
    if total_available < total_target:
        raise ValueError(f"总样本数不足: 可用 {total_available}，目标 {total_target}")

    exact = {k: (v / total_available) * total_target for k, v in counts.items()}
    allocated = {k: min(counts[k], math.floor(exact[k])) for k in counts}

    remaining = total_target - sum(allocated.values())
    while remaining > 0:
        candidates = [
            k
            for k in counts
            if allocated[k] < counts[k]
        ]
        if not candidates:
            break

        candidates.sort(key=lambda k: (exact[k] - math.floor(exact[k]), counts[k] - allocated[k]), reverse=True)

        for k in candidates:
            if remaining == 0:
                break
            if allocated[k] < counts[k]:
                allocated[k] += 1
                remaining -= 1

    if sum(allocated.values()) != total_target:
        raise RuntimeError("配额分配失败，未达到目标数量")

    return allocated


def main() -> int:
    parser = argparse.ArgumentParser(description="融合 raw_data 四个数据集并按比例抽样 500 条")
    parser.add_argument("--target", type=int, default=500, help="最终样本数量，默认 500")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("fused_qa_500.json")),
        help="输出文件路径",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "raw_data"

    dag_path = raw_dir / "dag_math_mini_processed.json"
    prism_path = raw_dir / "prism_physics_data_processed.json"
    hotpot_path = raw_dir / "HotpotQA.json"
    mmlu_path = raw_dir / "mmlu_pro_mini_280.json"

    dag_items = normalize_dag_math(load_json(dag_path))
    prism_items = normalize_prism(load_json(prism_path))
    hotpot_items = normalize_hotpot(load_json(hotpot_path))
    mmlu_items = normalize_mmlu(load_json(mmlu_path))

    datasets: Dict[str, List[Dict[str, str]]] = {
        "dag_math_mini": dag_items,
        "prism_physics": prism_items,
        "hotpotqa": hotpot_items,
        "mmlu_pro_mini": mmlu_items,
    }

    counts = {k: len(v) for k, v in datasets.items()}
    targets = proportional_counts(counts, args.target)

    rng = random.Random(args.seed)
    merged: List[Dict[str, str]] = []
    for name, items in datasets.items():
        k = targets[name]
        sampled = rng.sample(items, k) if k < len(items) else list(items)
        merged.extend(sampled)

    rng.shuffle(merged)

    for i, item in enumerate(merged):
        item["problem_id"] = str(i)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("融合完成")
    print(f"输出文件: {output_path}")
    print(f"总条数: {len(merged)}")
    print("各数据集抽样数量:")
    for name in datasets:
        print(f"  - {name}: {targets[name]} / {counts[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
