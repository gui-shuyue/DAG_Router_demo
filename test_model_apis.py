import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request

from model_config import MODELS_CONFIG


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def post_json(url: str, payload: Dict, headers: Dict[str, str], timeout: int) -> Tuple[int, Dict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"raw": body}
        return e.code, detail


def run_smoke_test(
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
    max_tokens: int,
    print_text: bool,
) -> Dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "LLM_DAG_Routing_02 smoke test",
    }

    started = time.time()
    status_code, result = post_json(url, payload, headers, timeout)
    latency = time.time() - started

    ok = status_code == 200 and isinstance(result, dict) and bool(result.get("choices"))

    output_text = ""
    usage = {}
    if ok:
        usage = result.get("usage") or {}
        output_text = (
            result["choices"][0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    return {
        "model": model,
        "ok": ok,
        "status": status_code,
        "latency_sec": latency,
        "text": output_text,
        "usage": usage,
        "error": None if ok else result,
    }


def estimate_cost(model: str, usage: Dict) -> Optional[float]:
    if model not in MODELS_CONFIG:
        return None

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0

    in_cost = MODELS_CONFIG[model]["input_cost"]
    out_cost = MODELS_CONFIG[model]["output_cost"]

    return (prompt_tokens / 1_000_000) * in_cost + (completion_tokens / 1_000_000) * out_cost


def parse_models(models_arg: Optional[str]) -> List[str]:
    if not models_arg:
        return list(MODELS_CONFIG.keys())
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test OpenRouter chat APIs for configured models")
    parser.add_argument("--models", help="Comma-separated model list. Default: all models in MODELS_CONFIG")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max completion tokens per request")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop immediately when one model fails")
    parser.add_argument("--print-text", action="store_true", help="Print returned text for successful requests")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    load_dotenv(workspace / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    if not api_key:
        print("[ERROR] Missing OPENROUTER_API_KEY. Set it in environment or .env.")
        return 2

    models = parse_models(args.models)
    if not models:
        print("[ERROR] No models to test.")
        return 2

    print(f"Testing {len(models)} model(s) via {base_url.rstrip('/')}/chat/completions")
    print("-" * 90)

    failures = 0

    for idx, model in enumerate(models, start=1):
        print(f"[{idx}/{len(models)}] {model} ...", flush=True)
        result = run_smoke_test(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            print_text=args.print_text,
        )

        if result["ok"]:
            usage = result["usage"]
            prompt_tokens = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")
            total_tokens = usage.get("total_tokens", "?")
            cost = estimate_cost(model, usage)
            cost_str = f"${cost:.8f}" if cost is not None else "N/A"

            print(
                f"  ✅ OK | status={result['status']} | latency={result['latency_sec']:.2f}s | "
                f"tokens(in/out/total)={prompt_tokens}/{completion_tokens}/{total_tokens} | est_cost={cost_str}"
            )
            if args.print_text and result["text"]:
                print(f"  ↳ text: {result['text']}")
        else:
            failures += 1
            print(
                f"  ❌ FAIL | status={result['status']} | latency={result['latency_sec']:.2f}s"
            )
            print(f"  ↳ error: {json.dumps(result['error'], ensure_ascii=False)}")
            if args.stop_on_fail:
                break

    print("-" * 90)
    if failures == 0:
        print("All model API checks passed.")
        return 0

    print(f"Completed with {failures} failure(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
