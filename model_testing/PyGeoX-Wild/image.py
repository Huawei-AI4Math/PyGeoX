import os
import json
import glob
import requests
import time
import re
import sys
import threading

# 重定向到文件时立即输出，避免 nohup/xxx.out 长时间无内容
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# ================= Configuration =================
API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")

MODEL_LIST = [
    "google/gemini-3-flash-preview"
]

# Project root (model_testing/PyGeoX-Wild/ -> repo root)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
INPUT_DIR = f"{PROJECT_ROOT}/model_testing/zhongkao_llm2/qwen3-rl-sparse/answer-output"
IMAGE_OUTPUT_DIR = f"{PROJECT_ROOT}/model_testing/zhongkao_llm2/qwen3-rl-sparse/image"
JSON_OUTPUT_DIR = f"{PROJECT_ROOT}/model_testing/zhongkao_llm2/qwen3-rl-sparse/image_info"
MAX_WORKERS = 5 

print_lock = threading.Lock()
# =================================================

def extract_python_code(text):
    """
    从 LLM 的回复中提取 python 代码块。
    """
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

def run_python_and_get_image(code, target_image_path):
    """
    在一个临时目录中运行生成的 python 代码，并捕获生成的 result.png。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "plot_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        try:
            # 执行脚本
            result = subprocess.run(
                ["python3", "plot_script.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            generated_img = os.path.join(tmpdir, "result.png")
            if os.path.exists(generated_img):
                shutil.copy(generated_img, target_image_path)
                return True, "Success"
            else:
                return False, f"Image not found. \nStdout: {result.stdout}\nStderr: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Execution timed out (30s)."
        except Exception as e:
            return False, str(e)

def fetch_completion(file_info):
    fname, problem_data, model_id, image_dir, json_dir = file_info
    # 基础路径（不带_error后缀）
    base_json_path = os.path.join(json_dir, fname)
    base_image_path = os.path.join(image_dir, fname.replace(".json", ".png"))
    
    # 如果 JSON 文件已存在（包括_error版本），跳过（断点续传）
    if os.path.exists(base_json_path) or os.path.exists(base_json_path.replace(".json", "_error.json")):
        return

    # 从 JSON 文件中读取 question 和 coordinate 字段
    desc = problem_data.get("question", "")
    solution = problem_data.get("coordinate", {})
    
    # 格式化 GT 坐标
    solution_text = ""
    if solution:
        solution_text = "\n\nGROUND TRUTH COORDINATES (Use these exact values in your matplotlib Points):\n"
        if "points" in solution:
            for p, coords in solution["points"].items():
                # 兼容 coords 为 [x,y] 或 单个 float/异常格式
                if hasattr(coords, "__len__") and len(coords) >= 2:
                    solution_text += f"Point {p}: ({coords[0]}, {coords[1]})\n"
                else:
                    solution_text += f"Point {p}: {coords}\n"
        if "circles" in solution:
            for c, r in solution["circles"].items():
                solution_text += f"Circle with center {c} has radius: {r}\n"

    system_prompt = (
        "You are a geometry visualization expert. Your task is to write a Python script using 'matplotlib' to render a geometric diagram.\n\n"
        "### CONSTRAINTS:\n"
        "1. USE MATPLOTLIB: Initialize points using `plt.scatter(x, y)` from the provided coordinates.\n"
        "2. DRAW RELATIONSHIPS: Based on the description, draw all Segments, Lines and Circles.\n"
        "3. STYLING: Use `ax.set_aspect('equal')` and `plt.axis('off')`.\n"
        "4. OUTPUT: The script MUST end with `plt.savefig('result.png')`.\n\n"
        "Return ONLY the code block starting with ```python."
    )
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Description: {desc}\n{solution_text}"}
    ]
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    max_attempts = 3
    
    # 代码执行重试循环
    for attempt in range(1, max_attempts + 1):
        # API 调用重试循环
        api_success = False
        full_text = ""
        py_code = ""
        last_api_error = "Unknown"
        
        for api_attempt in range(3):
            try:
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0
                }
                
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                     headers=headers, json=payload, timeout=120)
                
                if resp.status_code == 200:
                    full_text = resp.json()['choices'][0]['message']['content'].strip()
                    py_code = extract_python_code(full_text)
                    api_success = True
                    break
                else:
                    last_api_error = f"API returned status {resp.status_code}"
                    time.sleep(5)
            except Exception as e:
                last_api_error = str(e)
                time.sleep(2)
        
        if not api_success:
            # API 调用失败，记录并退出
            error_json_path = base_json_path.replace(".json", "_error.json")
            with open(error_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "item_id": fname,
                    "status": "api_failed",
                    "error": last_api_error,
                    "attempts": attempt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4, ensure_ascii=False)
            
            with print_lock:
                tqdm.write(f"❌ [API FAILED] {fname}")
            return
        
        # 执行代码并获取图片
        success, error_msg = run_python_and_get_image(py_code, base_image_path)
        
        if success:
            # 成功，保存结果
            result_data = {
                "item_id": fname,
                "generated_python_code": py_code,
                "execution_status": "success",
                "full_response": full_text,
                "attempts": attempt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(base_json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            
            with print_lock:
                tqdm.write(f"✅ [SUCCESS] {fname} (attempt {attempt})")
            return
        else:
            # 执行失败，准备重试
            if attempt < max_attempts:
                # 将错误信息添加到对话历史中
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "user",
                    "content": f"The previous code execution failed with the following error:\n\n{error_msg}\n\nPlease fix the error and generate a new complete Python script."
                })
                
                with print_lock:
                    tqdm.write(f"⚠️  [RETRY] {fname} (attempt {attempt}/{max_attempts}): {error_msg[:80]}...")
            else:
                # 最后一次尝试也失败了，保存错误信息
                error_json_path = base_json_path.replace(".json", "_error.json")
                result_data = {
                    "item_id": fname,
                    "generated_python_code": py_code,
                    "execution_status": "failed",
                    "execution_error": error_msg,
                    "full_response": full_text,
                    "attempts": max_attempts,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(error_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=4, ensure_ascii=False)
                
                with print_lock:
                    tqdm.write(f"❌ [FAILED] {fname} (after {max_attempts} attempts)")
                return

def main():
    # 创建输出目录
    Path(IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(JSON_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files:
        print(f"No files found in {INPUT_DIR}")
        return
    
    print(f"\n🚀 Starting Image Generation")
    print(f"   Images will be saved to: {IMAGE_OUTPUT_DIR}")
    print(f"   JSON info will be saved to: {JSON_OUTPUT_DIR}")
    
    tasks = []
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                problem_json = json.load(f)
                # 使用第一个 model（因为只有一个）
                model_id = MODEL_LIST[0]
                tasks.append((
                    os.path.basename(f_path), 
                    problem_json, 
                    model_id, 
                    IMAGE_OUTPUT_DIR,
                    JSON_OUTPUT_DIR
                ))
        except Exception as e:
            print(f"Error loading {f_path}: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(
            executor.map(fetch_completion, tasks), 
            total=len(tasks), 
            desc="Generating Images"
        ))

if __name__ == "__main__":
    main()