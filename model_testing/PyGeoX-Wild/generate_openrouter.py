import os
import json
import csv
import time
import re
import sys
import traceback
import warnings

# 重定向到文件时立即输出，避免 nohup/xxx.out 长时间无内容
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from typing import List, Dict, Optional

# --- 配置 --- (model_testing/PyGeoX-Wild/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = PROJECT_ROOT
ZHONGKAO_BENCH = BASE_PATH / "model_testing/zhongkao_llm2/zhongkao_english.csv"
PYGEOX_DOCS = BASE_PATH / "model_training/system_prompt_rl.md"

# --- 后端选择: OpenRouter 或 本地 vLLM ---
# 使用本地 vLLM 时设为 "1" 或 "true"，否则使用 OpenRouter
USE_VLLM = os.getenv("USE_VLLM", "1").lower() in ("1", "true", "yes")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://0.0.0.0:8893").rstrip("/")
VLLM_API_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

# OpenRouter 配置（仅当 USE_VLLM=False 时需要）
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
if not USE_VLLM and not OPENROUTER_API_KEY:
    print("警告: 未设置 OPENROUTER_API_KEY 环境变量")
    print("请设置: export OPENROUTER_API_KEY='your-api-key'")

# 模型选择
# 方式1: 一次测试多个模型，填写 MODEL_LIST（非空则依次跑完这些模型）
# 方式2: 只测一个模型时，把 MODEL_LIST 设为 []，使用环境变量 OPENROUTER_MODEL / VLLM_MODEL 或下面的 MODEL_NAME
# OpenRouter 模型列表: https://openrouter.ai/models
# 使用 vLLM 时 MODEL_NAME 可为任意标识（vLLM 单模型时通常忽略），便于区分输出目录
MODEL_LIST = [
    "deepseek/deepseek-v3.2",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview"
]
# 单模型时的默认
# 使用 vLLM 时：请求里的 model 必须与 vLLM 服务端一致（一般为 --model 的路径，见 serve_rl_model.py / vllm log 里的 served_model_name）
VLLM_DEFAULT_MODEL = str(BASE_PATH / "model_training/RL_checkpoint_sparse")
MODEL_NAME = os.getenv("VLLM_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"))
if USE_VLLM and not os.getenv("OPENROUTER_MODEL"):
    MODEL_NAME = os.getenv("VLLM_MODEL", VLLM_DEFAULT_MODEL)

# 输出文件夹 - 根据当前模型名称自动创建（多模型时在 main 里按模型切换）
model_slug = (MODEL_LIST[0] if MODEL_LIST else MODEL_NAME).replace("/", "-").replace(":", "-")
OUTPUT_FOLDER = BASE_PATH / f"model_testing/zhongkao_llm2/{model_slug}/answer-output"

# API 配置
# 使用 vLLM 时请求发往本地；否则使用 OpenRouter（必须使用完整 chat/completions 路径）
API_BASE_URL = VLLM_API_URL if USE_VLLM else "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300  # 5分钟超时
BATCH_SIZE = 1

# 推理参数
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192

# --- 核心辅助函数 ---

def load_system_prompt():
    """Load PyGeoX documentation as system prompt."""
    with open(PYGEOX_DOCS, "r", encoding="utf-8") as f:
        return f.read()

def create_user_prompt(question_text):
    """基于你提供的原始 Prompt 模板"""
    return f"""Write python code to find the coordinates and circle radiuses for:

Diagram description:
{question_text}

"""

def call_chat_api(messages: List[Dict], temperature: float = TEMPERATURE, top_p: float = TOP_P, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """
    调用 Chat Completions API 生成回复（支持 OpenRouter 或本地 vLLM）
    """
    headers = {"Content-Type": "application/json"}
    if not USE_VLLM and OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers["HTTP-Referer"] = "https://github.com/your-repo"
        headers["X-Title"] = "PyGeoX Geometry Solver"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_BASE_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            raw_text = response.text
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError:
                snippet = (raw_text or "(empty)")[:400].replace("\n", " ")
                try:
                    err_body = response.json()
                    print(f"  API HTTP 错误 (尝试 {attempt + 1}/{MAX_RETRIES}): status={response.status_code}, body={err_body}")
                except Exception:
                    print(f"  API HTTP 错误 (尝试 {attempt + 1}/{MAX_RETRIES}): status={response.status_code}, raw={snippet!r}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
                continue
            try:
                result = response.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                snippet = (raw_text or "(empty)")[:400].replace("\n", " ")
                print(f"  API 返回非 JSON (尝试 {attempt + 1}/{MAX_RETRIES}): status={response.status_code}, body={snippet!r}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
                continue
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            if "error" in result:
                print(f"  API 错误信息: {result.get('error', result)}")
                return None
            print(f"  API 响应格式异常: {result}")
            return None
        except requests.exceptions.Timeout:
            print(f"  API 请求超时 (尝试 {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"  API 请求错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception as e:
            print(f"  未知错误: {e}")
            traceback.print_exc()
            return None
    return None

def extract_python_code(completion):
    """Extract Python code from markdown code blocks."""
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if not matches:
        pattern = r'```\s*(.*?)```'
        matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        code = '\n\n'.join(matches)
        return code.strip()
    return None

def execute_code_and_check(code, problem_id):
    """执行代码并提取坐标和圆信息"""
    plt.close('all')
    namespace = {}
    setup_code = """
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exec(setup_code, namespace)
            exec(code, namespace)
        points = namespace.get('points', None)
        circles = namespace.get('circles', None)
        if points is None or not isinstance(points, dict):
            return False, "Missing or invalid 'points' dictionary."
        coordinate_data = {
            "points": points,
            "circles": circles if isinstance(circles, dict) else {}
        }
        return True, coordinate_data
    except Exception as e:
        error_msg = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg

def truncate_history_if_needed(history: List[Dict], max_tokens: int = 15000) -> List[Dict]:
    """截断过长的对话历史"""
    if len(history) <= 3:
        return history
    estimated_tokens = sum(len(msg.get("content", "")) for msg in history) // 4
    if estimated_tokens <= max_tokens:
        return history
    truncated = [history[0].copy(), history[1].copy()]
    if len(history) > 2:
        for msg in history[-2:]:
            msg_copy = msg.copy()
            if msg_copy["role"] == "assistant" and len(msg_copy.get("content", "")) > 5000:
                msg_copy["content"] = msg_copy["content"][:5000] + "\n\n[内容已截断...]"
            truncated.append(msg_copy)
    estimated_tokens_after = sum(len(msg.get("content", "")) for msg in truncated) // 4
    if estimated_tokens_after > max_tokens:
        for msg in truncated:
            if msg["role"] == "assistant" and len(msg.get("content", "")) > 3000:
                msg["content"] = msg["content"][:3000] + "\n\n[内容已截断...]"
    return truncated

def process_single_problem(problem: Dict, system_prompt: str, progress: Optional[str] = None):
    """处理单个问题的生成、执行与多轮重试逻辑"""
    problem_id = problem["id"]
    question_text = problem["question"]
    prefix = f"  [{progress}] " if progress else "  "
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": create_user_prompt(question_text)}
    ]
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        history = truncate_history_if_needed(history, max_tokens=12000)
        print(f"{prefix}问题 {problem_id} - 第 {attempt} 次尝试...")
        completion = call_chat_api(history, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)
        if completion is None:
            error_msg = "API 调用失败，无法获取模型回复。"
            if attempt < max_attempts:
                print(f"{prefix}× 问题 {problem_id} 失败: {error_msg} 准备重试")
                time.sleep(1)
                continue
            else:
                save_result(problem, "", None, attempt, error_msg, "failed")
                print(f"{prefix}✗ 问题 {problem_id} 在 {max_attempts} 次尝试后最终失败")
                return
        truncated_completion = completion[:8000] + ("\n\n[内容已截断...]" if len(completion) > 8000 else "")
        history.append({"role": "assistant", "content": truncated_completion})
        code = extract_python_code(completion)
        if code:
            success, result_info = execute_code_and_check(code, problem_id)
            if success:
                save_result(problem, completion, code, attempt, result_info, "success")
                print(f"{prefix}✓ 问题 {problem_id} 成功 (第 {attempt} 次尝试)")
                return
            else:
                error_msg = result_info
        else:
            error_msg = "未在模型回复中找到 Python 代码块。"
        if attempt < max_attempts:
            print(f"{prefix}× 问题 {problem_id} 失败: {error_msg[:100]}... 准备重试")
            history.append({
                "role": "user",
                "content": f"上一次运行出错，错误信息如下：\n\n{error_msg}\n\n请修复错误并重新生成完整的代码。"
            })
            time.sleep(1)
        else:
            save_result(problem, completion, code, attempt, error_msg, "failed")
            print(f"{prefix}✗ 问题 {problem_id} 在 {max_attempts} 次尝试后最终失败")

def convert_to_json_serializable(obj):
    """将 numpy/SymPy 等转为 JSON 可序列化"""
    import numpy as np
    try:
        from sympy import Basic, Integer, Float, Rational
        if isinstance(obj, (Integer, Float, Rational)):
            return float(obj)
        elif isinstance(obj, Basic):
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        try:
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

def save_result(item, completion, code, attempt, result_info, status):
    """将结果保存，包含 coordinate 字段"""
    suffix = "" if status == "success" else "_error"
    target_path = OUTPUT_FOLDER / f"problem_{item['id']}{suffix}.json"
    output_data = {
        "problem_id": item["id"],
        "question": item["question"],
        "status": status,
        "attempts": attempt,
        "model": MODEL_NAME,
        "coordinate": {},
        "execution_info": "success" if status == "success" else str(result_info),
        "extracted_code": code if code else "",
        "completion": completion,
    }
    if status == "success" and isinstance(result_info, dict):
        output_data["coordinate"] = convert_to_json_serializable(result_info)
    output_data = convert_to_json_serializable(output_data)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """主入口：读取 CSV，按模型列表依次测试（或单模型）"""
    global MODEL_NAME, OUTPUT_FOLDER
    from datetime import datetime
    print("\n" + "=" * 60, flush=True)
    print("  PyGeoX 中考几何 API 测试 (OpenRouter / vLLM) - 程序开始", flush=True)
    print("=" * 60, flush=True)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  题目 CSV: {ZHONGKAO_BENCH}", flush=True)
    print("=" * 60 + "\n", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if not USE_VLLM and not OPENROUTER_API_KEY:
        print("错误: 使用 OpenRouter 时请设置 OPENROUTER_API_KEY 环境变量", flush=True)
        print("例如: export OPENROUTER_API_KEY='YOUR_API_KEY'", flush=True)
        print("或使用本地 vLLM: export USE_VLLM=1 VLLM_BASE_URL=http://0.0.0.0:8893", flush=True)
        return
    if USE_VLLM:
        print(f"使用本地 vLLM: {VLLM_BASE_URL}", flush=True)
        # 使用 vLLM 时若未单独设置模型列表，则只跑当前 MODEL_NAME（避免对同一服务重复多模型）
        vllm_default_list = [
            "deepseek/deepseek-v3.2",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-3-flash-preview",
            "google/gemini-3-pro-preview",
        ]
        if MODEL_LIST == vllm_default_list:
            model_list = [MODEL_NAME]
            print(f"vLLM 单模型模式: {MODEL_NAME}", flush=True)
        else:
            model_list = MODEL_LIST if MODEL_LIST else [MODEL_NAME]
    elif MODEL_LIST:
        model_list = MODEL_LIST
        print(f"多模型模式: 共 {len(model_list)} 个模型", flush=True)
    else:
        model_list = [MODEL_NAME]
        print(f"单模型模式: {MODEL_NAME}", flush=True)
    system_prompt = load_system_prompt()
    all_problems = []
    if not ZHONGKAO_BENCH.exists():
        print(f"错误：找不到 CSV 文件 {ZHONGKAO_BENCH}")
        return
    with open(ZHONGKAO_BENCH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_problems.append({"id": row["id"], "question": row["question"]})
    print(f"题目总数: {len(all_problems)}\n", flush=True)
    sys.stdout.flush()
    for model_index, current_model in enumerate(model_list, 1):
        MODEL_NAME = current_model
        model_slug = MODEL_NAME.replace("/", "-").replace(":", "-")
        OUTPUT_FOLDER = BASE_PATH / f"model_testing/zhongkao_llm2/{model_slug}/answer-output"
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}", flush=True)
        print(f"模型 [{model_index}/{len(model_list)}]: {MODEL_NAME}", flush=True)
        print(f"输出目录: {OUTPUT_FOLDER}", flush=True)
        print("="*60, flush=True)
        pending_problems = [
            p for p in all_problems
            if not (OUTPUT_FOLDER / f"problem_{p['id']}.json").exists()
            and not (OUTPUT_FOLDER / f"problem_{p['id']}_error.json").exists()
        ]
        total_pending = len(pending_problems)
        print(f"总题目数: {len(all_problems)}, 待处理: {total_pending}")
        if not pending_problems:
            print("该模型下所有题目已处理完成，跳过。")
            continue
        model_start = time.time()
        for i, problem in enumerate(pending_problems, 1):
            pct = int(100 * i / total_pending) if total_pending else 0
            progress_tag = f"{i}/{total_pending}"
            print(f"\n--- [{pct:3d}%] 正在处理问题 {progress_tag}: {problem['id']} (模型 {model_index}/{len(model_list)}) ---")
            process_single_problem(problem, system_prompt, progress=progress_tag)
            if i < len(pending_problems):
                time.sleep(0.5)
        model_elapsed = time.time() - model_start
        n_success = sum(1 for p in all_problems if (OUTPUT_FOLDER / f"problem_{p['id']}.json").exists())
        n_fail = sum(1 for p in all_problems if (OUTPUT_FOLDER / f"problem_{p['id']}_error.json").exists())
        print(f"\n>>> 模型 [{model_index}/{len(model_list)}] {MODEL_NAME} 完成: 成功 {n_success}, 失败 {n_fail}, 共 {len(all_problems)} 题, 耗时 {model_elapsed:.1f}s")
    print("\n全部模型运行结束。", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    print("PyGeoX 中考几何 API 测试脚本启动 (OpenRouter / vLLM)", flush=True)
    sys.stdout.flush()
    main()
