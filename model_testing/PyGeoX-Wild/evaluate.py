#!/usr/bin/env python3
"""
使用外部视觉 LLM 评估：模型生成的几何图是否与题目描述一致。

评估结果二分类：完全符合题目（match=true）/ 不符合题目（match=false）。
若题目来源为带 error 的 JSON（如 problem_X_error.json），则不调用 API，自动归为不正确。

目录约定（与 generate / generate_openrouter 一致）：
  - {model_dir}/answer-output/problem_X.json 或 problem_X_error.json：含 problem_id, question
  - {model_dir}/image/problem_X.png：生成的图
  - {model_dir}/evaluation/problem_X.json：每题评估结果（含 match, reason）
  - {model_dir}/evaluation/stats.json：该模型单独统计（正确率、total、correct、details）

用法：
  python evaluate.py --model_dir qwen3-sft-sparse
  python evaluate.py --model_dir /path/to/model_dir --eval_model openai/gpt-4o
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# --- 配置 ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

# 模型输出目录（相对 zhongkao_llm2 或绝对路径），命令行可覆盖
MODEL_DIR = "qwen3-rl-sparse"

# 默认用于“看图判题”的视觉模型（需支持 vision）
DEFAULT_EVAL_MODEL = "google/gemini-3-flash-preview"

# 评估 prompt：二分类——完全符合题目 / 不符合题目，输出 JSON { "match": true/false, "reason": "..." }
EVAL_SYSTEM_PROMPT = """You are an expert in geometry diagrams. Given a diagram description (geometry problem statement) and an image of a generated diagram, judge whether the image FULLY matches the description (correct diagram for this problem) or NOT.

Output ONLY a single JSON object, no other text, with exactly two keys:
- "match": boolean. true = the image fully matches the description (完全符合题目); false = the image does not match or is incomplete/wrong (不符合题目).
- "reason": short explanation in English.

Be strict: only set "match" to true when the diagram correctly shows all required shapes (circles, lines, points), labels, and relative positions stated in the problem. Otherwise set "match" to false."""

EVAL_USER_TEMPLATE = """Diagram description (geometry problem):
{question}

Look at the attached image. Does it FULLY match the description above? Reply with only the JSON object: {{"match": true or false, "reason": "..."}}."""


def load_answer_output(model_dir: Path) -> dict[str, dict]:
    """从 answer-output 目录加载所有题目：problem_id -> { question, ... }。"""
    out = {}
    ao = model_dir / "answer-output"
    if not ao.is_dir():
        return out
    # 先收集所有文件，同一 pid 优先用非 _error 的 JSON
    files_by_pid: dict[str, list[Path]] = {}
    for f in ao.iterdir():
        if f.suffix != ".json":
            continue
        name = f.stem  # problem_1 或 problem_1_error
        if name.endswith("_error"):
            pid = name.replace("_error", "").replace("problem_", "")
        else:
            pid = name.replace("problem_", "")
        if not pid.isdigit():
            continue
        files_by_pid.setdefault(pid, []).append(f)
    for pid, flist in files_by_pid.items():
        # 优先非 _error
        flist.sort(key=lambda p: ("_error" in p.stem, p.name))
        for f in flist:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            question = data.get("question") or data.get("Diagram description") or ""
            if not question and "completion" in data:
                question = data.get("problem_id", pid)  # 仅作占位
            is_error = f.stem.endswith("_error")
            out[pid] = {"problem_id": pid, "question": question, "source_file": f.name, "is_error": is_error}
            break
    return out


def list_problems_with_images(model_dir: Path) -> list[tuple[str, str, bool]]:
    """返回 (problem_id, question, is_error) 且对应 image 存在的题目列表。题目来源带 error 则 is_error=True，无需调用 API，自动归为不正确。"""
    problems = load_answer_output(model_dir)
    image_dir = model_dir / "image"
    result = []
    for pid, info in problems.items():
        if not info.get("question"):
            continue
        img_path = image_dir / f"problem_{pid}.png"
        if img_path.is_file():
            result.append((pid, info["question"], info.get("is_error", False)))
    return result


def image_to_base64_data_url(path: Path) -> str:
    """将本地图片转为 data URL（OpenRouter 要求的 image_url）。"""
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_vision_api(question: str, image_base64_url: str, model: str) -> str | None:
    """调用 OpenRouter 视觉模型，返回回复文本。"""
    if not OPENROUTER_API_KEY:
        print("错误: 请设置环境变量 OPENROUTER_API_KEY", file=sys.stderr)
        return None
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": EVAL_USER_TEMPLATE.format(question=question)},
                {"type": "image_url", "image_url": {"url": image_base64_url}},
            ],
        },
    ]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PyGeoX",
        "X-Title": "PyGeoX Diagram Evaluation",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"].get("content")
            if "error" in data:
                print(f"  API error: {data['error']}", file=sys.stderr)
            break
        except requests.exceptions.RequestException as e:
            print(f"  请求异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def parse_match_reason(text: str) -> tuple[bool | None, str]:
    """从 LLM 回复中解析 match 和 reason；若无法解析则返回 (None, raw_text)。"""
    if not text:
        return None, ""
    text = text.strip()
    # 尝试从 markdown 代码块中取出 JSON
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    # 尝试直接解析整段为 JSON
    try:
        obj = json.loads(text)
        match = obj.get("match")
        if match is not None:
            match = bool(match)
        return match, (obj.get("reason") or "")
    except (json.JSONDecodeError, ValueError):
        pass
    # 回退：尝试找 "match": true/false
    sm = re.search(r'"match"\s*:\s*(true|false)', text, re.IGNORECASE)
    match = sm.group(1).lower() == "true" if sm else None
    rm = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    reason = (rm.group(1).replace("\\\"", "\"").strip() if rm else text)
    return match, reason


def main():
    parser = argparse.ArgumentParser(description="用外部视觉 LLM 评估生成图与题目描述是否一致")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=MODEL_DIR,
        help=f"模型输出目录名（相对 zhongkao_llm2）或绝对路径，默认: {MODEL_DIR}",
    )
    parser.add_argument(
        "--eval_model",
        default=DEFAULT_EVAL_MODEL,
        help=f"用于评估的 OpenRouter 视觉模型 (默认: {DEFAULT_EVAL_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多评估题目数，0 表示不限制",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若 evaluation 中已有该题结果则跳过",
    )
    args = parser.parse_args()

    # 解析 model_dir
    raw = Path(args.model_dir)
    if raw.is_absolute():
        model_dir = raw
    else:
        model_dir = Path(__file__).resolve().parent / raw
    if not model_dir.is_dir():
        print(f"错误: 目录不存在 {model_dir}", file=sys.stderr)
        sys.exit(1)

    eval_dir = model_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_problems_with_images(model_dir)
    if not pairs:
        print(f"未找到同时具备 answer-output 与 image 的题目（目录: {model_dir}）")
        return

    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"共 {len(pairs)} 题待评估，评估模型: {args.eval_model}")

    for i, (pid, question, is_error) in enumerate(pairs, 1):
        out_path = eval_dir / f"problem_{pid}.json"
        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(pairs)}] 跳过 problem_{pid}（已存在）")
            continue

        img_path = model_dir / "image" / f"problem_{pid}.png"
        if not img_path.is_file():
            continue

        # 题目来源带 error：不调用 API，自动归为不正确
        if is_error:
            print(f"[{i}/{len(pairs)}] problem_{pid} 来源带 error，自动归为不正确")
            record = {
                "problem_id": pid,
                "question": question,
                "match": False,
                "reason": "题目名称带error，自动归为不正确",
                "auto_error": True,
                "evaluator_model": None,
            }
        else:
            b64_url = image_to_base64_data_url(img_path)
            print(f"[{i}/{len(pairs)}] 评估 problem_{pid} ...", end=" ", flush=True)
            raw_response = call_vision_api(question, b64_url, args.eval_model)
            if raw_response is None:
                print("API 调用失败")
                record = {
                    "problem_id": pid,
                    "question": question,
                    "match": None,
                    "reason": "",
                    "error": "api_failed",
                    "evaluator_model": args.eval_model,
                }
            else:
                match, reason = parse_match_reason(raw_response)
                print(f"match={match} reason={reason[:50]}..." if reason and len(reason) > 50 else f"match={match} reason={reason}")
                record = {
                    "problem_id": pid,
                    "question": question,
                    "match": match,
                    "reason": reason,
                    "raw_response": raw_response,
                    "evaluator_model": args.eval_model,
                }
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # 汇总并写入该模型单独的统计文件
    total = 0
    correct = 0
    details = []
    for j in eval_dir.glob("problem_*.json"):
        if j.name == "stats.json":
            continue
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
            total += 1
            if data.get("match") is True:
                correct += 1
            details.append({"problem_id": data.get("problem_id"), "match": data.get("match")})
        except (json.JSONDecodeError, OSError):
            pass
    accuracy = (correct / total) if total else 0.0
    stats = {
        "model_dir": str(model_dir.name),
        "total": total,
        "correct": correct,
        "正确率": round(accuracy, 4),
        "accuracy": round(accuracy, 4),
        "evaluator_model": args.eval_model,
        "details": details,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats_path = eval_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n已评估 {total} 题，正确 {correct} 题，正确率: {accuracy:.2%}")
    print(f"统计已写入: {stats_path}")


if __name__ == "__main__":
    main()
