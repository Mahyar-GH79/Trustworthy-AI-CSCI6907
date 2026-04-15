#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.4"
PROMPTS_PATH = Path("prompts.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "responses.json"


def extract_output_text(payload):
    texts = []

    def walk(node):
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "output_text" and isinstance(node.get("text"), str):
                texts.append(node["text"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return "\n".join(part.strip() for part in texts if part.strip()).strip()


def load_prompts():
    with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def call_openai(api_key, model, prompt):
    body = {
        "model": model,
        "input": prompt,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("Missing OPENAI_API_KEY environment variable.", file=sys.stderr)
        return 1

    prompts = load_prompts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model,
        "api": "responses",
        "created_at": int(time.time()),
        "entries": [],
    }

    for item in prompts:
        entry_id = item["entry_id"]
        print(f"Running entry {entry_id}/30: {item['prompt_goal']}", file=sys.stderr)
        try:
            raw_response = call_openai(api_key, model, item["prompt"])
            text_response = extract_output_text(raw_response)
            result = {
                "entry_id": entry_id,
                "task_group": item["task_group"],
                "research_task": item["research_task"],
                "prompt_goal": item["prompt_goal"],
                "prompt": item["prompt"],
                "response_id": raw_response.get("id"),
                "status": raw_response.get("status"),
                "output_text": text_response,
                "raw_response": raw_response,
            }
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            result = {
                "entry_id": entry_id,
                "task_group": item["task_group"],
                "research_task": item["research_task"],
                "prompt_goal": item["prompt_goal"],
                "prompt": item["prompt"],
                "error": {
                    "type": "http_error",
                    "status_code": exc.code,
                    "body": error_body,
                },
            }
        except Exception as exc:
            result = {
                "entry_id": entry_id,
                "task_group": item["task_group"],
                "research_task": item["research_task"],
                "prompt_goal": item["prompt_goal"],
                "prompt": item["prompt"],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

        results["entries"].append(result)
        OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        time.sleep(0.2)

    print(f"Saved results to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
