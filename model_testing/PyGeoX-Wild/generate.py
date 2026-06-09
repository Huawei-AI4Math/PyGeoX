#!/usr/bin/env python3
"""
GGBench 测试脚本

⚠️  重要：为了强制使用 vLLM v0 引擎，必须在命令行中设置环境变量！

正确启动方式：

    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 nohup python generate.py > nohup.out 2>&1 &

或者：

    export CUDA_VISIBLE_DEVICES=0
    export VLLM_USE_V1=0
    nohup python generate.py > nohup.out 2>&1 &

注意：VLLM_USE_V1 环境变量必须在 Python 启动之前设置，在代码中设置无效！
vLLM 0.10.2 在导入时就会决定使用哪个引擎，因此环境变量必须在导入之前设置。
"""

import os
import sys

# 检查环境变量是否在命令行中设置
vllm_use_v1 = os.environ.get("VLLM_USE_V1", "")
if vllm_use_v1 != "0":
    print("=" * 80)
    print("⚠️  警告: VLLM_USE_V1 环境变量未在命令行中设置为 '0'")
    print(f"   当前值: VLLM_USE_V1={vllm_use_v1 if vllm_use_v1 else '(未设置)'}")
    print("   这可能导致 vLLM 使用 v1 引擎（可能遇到 CUDA 初始化问题）")
    print("   请在命令行中设置: VLLM_USE_V1=0 python generate.py")
    print("=" * 80)
    # 尝试在代码中设置（可能无效，但至少尝试一下）
    os.environ["VLLM_USE_V1"] = "0"
    print("   已在代码中设置 VLLM_USE_V1=0（可能无效，建议在命令行中设置）\n")
else:
    print(f"✓ 检测到命令行设置的 VLLM_USE_V1=0（正确）\n")

# 必须在任何其他导入之前设置环境变量
# 指定 GPU - 必须在导入任何可能使用 CUDA 的库之前设置
target_gpu = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu

# 验证环境变量已设置
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("错误: CUDA_VISIBLE_DEVICES 未设置")
    sys.exit(1)

print(f"✓ 设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"✓ 目标 GPU: {target_gpu} (在 CUDA_VISIBLE_DEVICES 设置后，将显示为设备 0)")
print(f"✓ 强制使用 vLLM v0 引擎 (VLLM_USE_V1=0)")

# 先导入基础库（不涉及 CUDA）
import json
import csv
import time
import re
import traceback
from pathlib import Path

# 延迟导入可能触发 CUDA 的库
# 先设置 matplotlib（使用非交互式后端，避免可能的 CUDA 初始化）
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt

# 现在导入可能使用 CUDA 的库
try:
    # 注意：vllm 在导入时可能会尝试初始化 CUDA
    # 如果出现错误，可能需要检查 CUDA 驱动或环境
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
except ImportError as e:
    print(f"请安装依赖: pip install vllm transformers flash_attn")
    print(f"导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"导入库时发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 简化 CUDA 检查 - 避免在初始化阶段调用可能失败的 CUDA 函数
# 让 vLLM 自己处理 CUDA 初始化，它会更稳健
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print("注意: 跳过预检查，让 vLLM 直接处理 CUDA 初始化（更稳健）")

# --- 配置 --- (model_testing/PyGeoX-Wild/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_PATH = PROJECT_ROOT
ZHONGKAO_BENCH = BASE_PATH / "model_testing/zhongkao_llm2/zhongkao_english.csv"
PYGEOX_DOCS = BASE_PATH / "model_training/system_prompt_rl.md"
OUTPUT_FOLDER = BASE_PATH / "model_testing/zhongkao_llm2/qwen3-rl-sparse/answer-output"

# 模型本地路径
MODEL_PATH = f"{PROJECT_ROOT}/model_training/merged/qwen3-8b-RL-sparse-step30"
GPU_COUNT = 1 
BATCH_SIZE = 16  # H200 显存很大，建议开启 Batch 提高稳定性

# ========== 推理质量优化配置 ==========
# 如果推理质量不如 API，可以尝试以下调整：
USE_FP8_QUANTIZATION = False  # True: 启用 FP8 量化以节省显存（当前显存不足，必须启用）
TEMPERATURE = 0.3  # 降低温度（0.1-0.5）获得更确定性的结果，提高温度（0.7-1.0）获得更多样性
TOP_P = 0.95  # 提高 top_p (0.9-0.99) 通常能提升质量
TOP_K = 50  # 限制候选 token 数量，None 表示不限制
REPETITION_PENALTY = 1.1  # 1.05-1.15 之间，防止重复
USE_FIXED_SEED = True  # True: 使用固定种子（可复现），False: 随机（更接近 API 的多样性）
SEED_VALUE = 42  # 固定种子值
# =====================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# --- 初始化 vLLM 与 Tokenizer ---
print(f"正在加载模型: {MODEL_PATH}")
print(f"推理配置: FP8量化={USE_FP8_QUANTIZATION}, Temperature={TEMPERATURE}, Top_p={TOP_P}")
print(f"✓ 使用 vLLM v0 引擎 (VLLM_USE_V1={os.environ.get('VLLM_USE_V1', '未设置')})")
print("注意: 让 vLLM 自己处理 CUDA 初始化（避免初始化顺序问题）")

# 构建 LLM 初始化参数
# 根据实际可用显存（约1.33 GiB，另一个进程占用103.24 GiB）调整参数
llm_kwargs = {
    "model": MODEL_PATH,
    "tensor_parallel_size": GPU_COUNT,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.40,  # 进一步降低到 0.20 以适应极少的可用显存
    "max_model_len": 16384,  # 降低到 16384 以节省显存
    "max_num_seqs": 256,  # 限制并发序列数，减少warmup时的显存需求（默认1024太大）
    "enable_prefix_caching": True,
    "swap_space": 8,  # 增加 swap space 以应对显存不足
}

# 根据配置选择精度
if USE_FP8_QUANTIZATION:
    llm_kwargs["kv_cache_dtype"] = "fp8"
    print("⚠️  使用 FP8 量化（可能降低推理质量，但节省显存）")
else:
    llm_kwargs["dtype"] = "auto"  # 自动选择最佳精度（通常是 bf16/fp16）
    print("✓ 使用自动精度（推荐，质量更好）")

# 初始化 vLLM，让它在自己的进程中处理 CUDA 初始化
print("正在初始化 vLLM 引擎（这可能需要一些时间）...")
try:
    llm = LLM(**llm_kwargs)
    print("✓ vLLM 引擎初始化成功")
except RuntimeError as e:
    if "CUDA" in str(e) or "cuda" in str(e) or "802" in str(e):
        print(f"\n❌ CUDA 初始化错误: {e}")
        print("\n可能的解决方案:")
        print("1. 确保在命令行中设置环境变量: CUDA_VISIBLE_DEVICES=0 python generate.py")
        print("2. 检查 nvidia-smi 是否能正常运行")
        print("3. 检查是否有其他进程占用 GPU")
        print("4. 尝试重启 Python 进程或系统")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"\n❌ vLLM 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 构建采样参数
# 收集 stop token IDs，过滤掉 None 值
stop_token_ids = []
if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if im_end_id is not None:
    stop_token_ids.append(im_end_id)

sampling_kwargs = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.05,  # 建议稍微给一点 min_p 过滤长尾噪声
    "repetition_penalty": 1.05,
    "max_tokens": 8192,  # 几何推理建议给长一点空间
    "skip_special_tokens": False,  # 保持特殊 token 以匹配 API 行为
}

# 只在有stop token时添加该参数
if stop_token_ids:
    sampling_kwargs["stop_token_ids"] = stop_token_ids

if USE_FIXED_SEED:
    sampling_kwargs["seed"] = SEED_VALUE
    print(f"✓ 使用固定种子: {SEED_VALUE}（可复现）")
else:
    print("✓ 使用随机种子（更接近 API 的多样性）")

sampling_params = SamplingParams(**sampling_kwargs)

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

def extract_python_code(completion):
    """Extract Python code from markdown code blocks."""
    # Pattern to match ```python ... ```
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, completion, re.DOTALL)
    
    if not matches:
        # Try without language specification
        pattern = r'```\s*(.*?)```'
        matches = re.findall(pattern, completion, re.DOTALL)
    
    if matches:
        # Combine all code blocks
        code = '\n\n'.join(matches)
        return code.strip()
    
    return None

def execute_code_and_check(code, problem_id):
    """执行代码并提取坐标和圆信息"""
    plt.close('all')
    
    # 准备执行环境
    namespace = {}
    setup_code = """
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 如果你的 PyGeoX 库有特定导入需求，也可以加在这里
"""
    
    try:
        # 1. 注入基础库并执行代码
        exec(setup_code, namespace)
        exec(code, namespace)
        
        # 2. 提取关键变量
        points = namespace.get('points', None)
        circles = namespace.get('circles', None)
        
        # 3. 验证数据有效性
        if points is None or not isinstance(points, dict):
            return False, "Missing or invalid 'points' dictionary."
        
        # 构造最终的 coordinate 字典
        coordinate_data = {
            "points": points,
            "circles": circles if isinstance(circles, dict) else {}
        }
        
        return True, coordinate_data
        
    except Exception as e:
        error_msg = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg

def truncate_history_if_needed(history, max_tokens=12000):
    """
    如果对话历史太长，截断以保留最重要的部分：
    - 保留 system prompt
    - 保留原始 user prompt
    - 只保留最近的 1-2 轮对话（assistant + user），并截断过长的 assistant 回复
    """
    if len(history) <= 3:  # system + user + 最多1轮，不需要截断
        return history
    
    # 检查 token 长度
    try:
        full_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer.encode(full_prompt))
        
        if token_count <= max_tokens:
            return history
        
        # 如果太长，只保留：system + 原始user + 最近的1轮对话（assistant + user）
        truncated = [history[0].copy()]  # system prompt
        truncated.append(history[1].copy())  # 原始 user prompt
        
        # 保留最后2条消息（assistant + user），如果有的话
        if len(history) > 2:
            for msg in history[-2:]:
                msg_copy = msg.copy()
                # 如果 assistant 回复太长，截断到前5000个字符
                if msg_copy["role"] == "assistant" and len(msg_copy["content"]) > 5000:
                    msg_copy["content"] = msg_copy["content"][:5000] + "\n\n[内容已截断...]"
                truncated.append(msg_copy)
        
        # 再次检查截断后的长度
        truncated_prompt = tokenizer.apply_chat_template(truncated, tokenize=False, add_generation_prompt=True)
        truncated_token_count = len(tokenizer.encode(truncated_prompt))
        
        # 如果还是太长，进一步截断 assistant 内容
        if truncated_token_count > max_tokens:
            for msg in truncated:
                if msg["role"] == "assistant" and len(msg["content"]) > 3000:
                    msg["content"] = msg["content"][:3000] + "\n\n[内容已截断...]"
        
        return truncated
    except Exception:
        # 如果检查失败，保守地只保留前3条
        if len(history) > 3:
            return [history[0], history[1]] + history[-2:] if len(history) > 2 else history[:3]
        return history
    
def process_batch(batch, system_prompt):
    """处理一批数据的生成、执行与多轮重试逻辑"""
    # 1. 初始化每个问题的对话历史
    batch_histories = []
    for item in batch:
        batch_histories.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": create_user_prompt(item["question"])}
        ])
    
    # 跟踪活跃（尚未成功且未耗尽尝试次数）的索引
    active_indices = list(range(len(batch)))
    max_attempts = 5 

    for attempt in range(1, max_attempts + 1):
        if not active_indices:
            break
        
        # 在生成前截断过长的对话历史（留出4000 token的余量用于生成）
        for idx in active_indices:
            batch_histories[idx] = truncate_history_if_needed(batch_histories[idx], max_tokens=12000)
        
        # 构造当前活跃项的 Prompts
        current_prompts = [
            tokenizer.apply_chat_template(batch_histories[idx], tokenize=False, add_generation_prompt=True)
            for idx in active_indices
        ]
        
        # vLLM 批量推理
        outputs = llm.generate(current_prompts, sampling_params)
        
        next_active_indices = []
        for i, idx in enumerate(active_indices):
            problem_id = batch[idx]["id"]
            question_text = batch[idx]["question"]
            completion = outputs[i].outputs[0].text
            code = extract_python_code(completion)
            
            # 将模型回答记录到历史中以备后续可能的重试
            # 限制 assistant 回复长度，避免历史过长（保留前8000字符）
            truncated_completion = completion[:8000] + ("\n\n[内容已截断...]" if len(completion) > 8000 else "")
            batch_histories[idx].append({"role": "assistant", "content": truncated_completion})

            if code:
                success, result_info = execute_code_and_check(code, problem_id)
                if success:
                    save_result(batch[idx], completion, code, attempt, result_info, "success")
                    print(f"✓ 问题 {problem_id} 成功 (第 {attempt} 次尝试)")
                    continue
                else:
                    error_msg = result_info
            else:
                error_msg = "未在模型回复中找到 Python 代码块。"

            # 失败且仍有重试机会时，更新对话历史
            if attempt < max_attempts:
                print(f"  × 问题 {problem_id} 失败: {error_msg[:100]}... 准备重试")
                batch_histories[idx].append({
                    "role": "user", 
                    "content": f"上一次运行出错，错误信息如下：\n\n{error_msg}\n\n请修复错误并重新生成完整的代码。"
                })
                next_active_indices.append(idx)
            else:
                # 最终失败保存
                save_result(batch[idx], completion, code, attempt, error_msg, "failed")
                print(f"✗ 问题 {problem_id} 在 {max_attempts} 次尝试后最终失败")

        active_indices = next_active_indices

def convert_to_json_serializable(obj):
    """将 numpy 数组和其他不可序列化的对象转换为 JSON 可序列化的格式"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj.__class__.__name__ == "Rational":
        # 处理 sympy 或其他库的 Rational 类型，转换为 float
        try:
            return float(obj)
        except (TypeError, ValueError):
            # 如果无法转换为 float，尝试转换为字符串
            return str(obj)
    elif hasattr(obj, '__float__'):
        # 处理其他可以转换为 float 的数值类型
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)
    else:
        return obj

def save_result(item, completion, code, attempt, result_info, status):
    """将结果保存，包含 coordinate 字段"""
    suffix = "" if status == "success" else "_error"
    target_path = OUTPUT_FOLDER / f"problem_{item['id']}{suffix}.json"
    
    output_data = {
        "problem_id": item["id"],
        "question": item["question"],
        "status": status,
        "attempts": attempt,
        "coordinate": {},  # 默认空
        "execution_info": "success" if status == "success" else str(result_info),
        "extracted_code": code if code else "",  # 如果code为None，设为空字符串
        "completion": completion,
    }
    
    if status == "success" and isinstance(result_info, dict):
        # 将提取到的 points 和 circles 存入 coordinate 键
        # 转换 numpy 数组为列表以确保 JSON 可序列化
        output_data["coordinate"] = convert_to_json_serializable(result_info)
    
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """主入口：读取 CSV 并触发批量处理"""
    system_prompt = load_system_prompt()
    all_problems = []
    
    if not ZHONGKAO_BENCH.exists():
        print(f"错误：找不到 CSV 文件 {ZHONGKAO_BENCH}")
        return

    with open(ZHONGKAO_BENCH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_problems.append({
                "id": row["id"], 
                "question": row["question"],
                "difficulty": row.get("difficulty", "")
            })

    print(f"测试数据集总题目数: {len(all_problems)} 道")

    # 过滤已经处理过的题目（跳过已存在的 .json 或 _error.json 文件）
    pending_problems = [
        p for p in all_problems 
        if not (OUTPUT_FOLDER / f"problem_{p['id']}.json").exists()
        and not (OUTPUT_FOLDER / f"problem_{p['id']}_error.json").exists()
    ]
    
    print(f"总题目数: {len(all_problems)}, 待处理: {len(pending_problems)}")

    # 分批次调用 vLLM
    for i in range(0, len(pending_problems), BATCH_SIZE):
        batch = pending_problems[i : i + BATCH_SIZE]
        print(f"\n--- 正在处理 Batch {i//BATCH_SIZE + 1} ({len(batch)} 个问题) ---")
        process_batch(batch, system_prompt)

if __name__ == "__main__":
    main()