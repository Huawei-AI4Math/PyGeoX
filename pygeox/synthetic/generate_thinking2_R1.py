"""
Generate and write R1 artifacts into existing `json_processed_2` problems (in-place).

What this does
- Iterates all JSON files in `generated_data_penalty/json_processed_2`
- Uses 10 parallel workers (default) to call an OpenRouter R1 model
- Builds a prompt exactly like the snippet in `model_training/prepare_sft_data.ipynb`
  (general thinking prompt + nl_description + ground-truth JSON schema)
- Prompts R1 to SOLVE the problem (does NOT reuse any prior generated answer/verify)
- Saves R1 reasoning/thinking (if returned) + full request/response metadata into:
    data["R1_approach"] = {"artifact": artifact_dict}
- Writes the modified JSON back to the original file atomically.

Environment
- API_KEY (or OPENROUTER_API_KEY)
- OPEN_AI_BASE_URL (default: https://openrouter.ai/api/v1)
- OPENROUTER_R1_MODEL (default: deepseek/deepseek-r1)

Notes
- This script is intentionally robust: it retries on network errors, parse errors,
  and missing/empty fields in the model output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "generated_data_penalty/json_processed_2_fixed"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        prefix=path.name + ".tmp.",
        suffix=".json",
    ) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    return json.loads(_strip_code_fences(text))


def _verify_has_score(verify: str) -> bool:
    v = verify.strip()
    return "score" in v.lower()


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def hide_numerical_values(obj: Any) -> Any:
    """Recursively replace numerical values with None while preserving structure."""
    if isinstance(obj, dict):
        return {k: hide_numerical_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [
            None if isinstance(item, (int, float)) else hide_numerical_values(item)
            for item in obj
        ]
    if isinstance(obj, (int, float)):
        return None
    return obj


def get_ground_truth_json(data: Dict[str, Any]) -> str:
    """
    Construct a ground-truth *schema* JSON string (numbers hidden as null).

    Preference order:
    - data["ground_truth"] (if present)
    - data["possible_solution"] (common in json_processed_2)
    - data["points"]/data["circles"] (fallback)
    """
    if isinstance(data.get("ground_truth"), dict):
        hidden = hide_numerical_values(data["ground_truth"])
        return json.dumps(hidden, indent=2, ensure_ascii=False)

    if isinstance(data.get("possible_solution"), dict):
        hidden = hide_numerical_values(data["possible_solution"])
        # Ensure keys exist at least.
        if isinstance(hidden, dict):
            hidden.setdefault("points", {})
            hidden.setdefault("circles", {})
        return json.dumps(hidden, indent=2, ensure_ascii=False)

    ground_truth: Dict[str, Any] = {"points": {}, "circles": {}}
    if isinstance(data.get("points"), dict):
        ground_truth["points"] = data["points"]
    if isinstance(data.get("circles"), dict):
        ground_truth["circles"] = data["circles"]

    hidden = hide_numerical_values(ground_truth)
    return json.dumps(hidden, indent=2, ensure_ascii=False)


@dataclass
class R1Result:
    ok: bool
    artifact: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def build_r1_prompt(system_prompt: str, nl_description: str, ground_truth_json: str) -> str:
    """
    Match the prompt template from `prepare_sft_data.ipynb` (lines 18–35).
    """
    return f"""
# Instruction
{system_prompt}

# INPUT
{nl_description}

#  ANSWER FORMAT:
```json
{ground_truth_json}
```

Please solve this problem and provide your response in the required JSON format.
""".lstrip()


def _extract_reasoning_from_message(msg: Any) -> str:
    """
    Best-effort extraction of provider reasoning.

    Notes:
    - OpenAI SDK message types won't necessarily have a `.reasoning` attribute.
      Provider-specific fields can be stored in pydantic `model_extra`.
    """
    # Direct attributes (DeepSeek native API exposes `reasoning_content` here)
    for key in ("reasoning_content", "reasoning", "thinking", "analysis"):
        try:
            val = getattr(msg, key, None)
            if isinstance(val, str) and val.strip():
                return val
        except Exception:
            pass

    # Pydantic extra fields (common for OpenRouter / provider passthrough)
    try:
        extra = getattr(msg, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("reasoning", "thinking", "analysis", "reasoning_content"):
                val = extra.get(key)
                if isinstance(val, str) and val.strip():
                    return val
    except Exception:
        pass

    return ""


def _extract_reasoning_from_response_dump(resp_dump: Dict[str, Any]) -> str:
    """
    Fallback extraction from the full response dict.
    This is more robust across provider/model variations.
    """
    try:
        choices = resp_dump.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            if isinstance(msg, dict):
                for key in ("reasoning", "thinking", "analysis", "reasoning_content"):
                    val = msg.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
    except Exception:
        pass
    return ""


def _call_r1_solve(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tries: int,
    save_failures_to: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> R1Result:
    last_err = None
    for attempt in range(max_tries):
        ts = _utc_ts()
        try:
            create_kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "PyGeoX"),
                },
            )

            # Prefer strict JSON + include_reasoning if supported; otherwise fallback.
            try:
                resp = client.chat.completions.create(
                    **create_kwargs,
                    extra_body={"include_reasoning": True},
                )
            except TypeError:
                resp = client.chat.completions.create(**create_kwargs)

            msg = resp.choices[0].message
            resp_dump = resp.model_dump()
            content = (msg.content or "").strip()
            
            # Log response metadata for debugging
            logger.debug(f"Attempt {attempt + 1}/{max_tries}: Response received, content length: {len(content)}")
            
            # Check if content is empty before parsing
            if not content:
                # Log detailed failure info
                logger.warning(
                    f"Empty content in model response. "
                    f"Raw content type: {type(msg.content)}, "
                    f"Raw content length: {len(msg.content or '')}, "
                    f"Has choices: {len(resp.choices) > 0}, "
                    f"Response keys: {list(resp_dump.keys()) if isinstance(resp_dump, dict) else 'N/A'}"
                )
                raise ValueError(f"Empty content in model response (content length: {len(msg.content or '')})")
            
            try:
                parsed = _parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as e:
                # Include a snippet of the content for debugging
                content_preview = content[:500] if len(content) > 500 else content
                logger.warning(
                    f"JSON parsing failed. Error: {str(e)}. "
                    f"Content length: {len(content)}, "
                    f"Content preview (first 500 chars): {repr(content_preview)}"
                )
                raise ValueError(f"Failed to parse JSON from model response: {str(e)}. Content preview: {repr(content_preview)}")

            answer_out = parsed.get("answer", "")
            verify_out = parsed.get("verify", "")

            # Normalize answer if it's an object.
            if isinstance(answer_out, (dict, list)):
                answer_out = json.dumps(answer_out, ensure_ascii=False)
            if not isinstance(verify_out, str):
                verify_out = str(verify_out)

            if not _is_nonempty_str(answer_out) or not _is_nonempty_str(verify_out):
                logger.warning(
                    f"Empty answer or verify. "
                    f"Answer type: {type(answer_out)}, Answer length: {len(str(answer_out))}, "
                    f"Verify type: {type(verify_out)}, Verify length: {len(str(verify_out))}"
                )
                raise ValueError("Empty 'answer' or 'verify' in model output")
            if not _verify_has_score(verify_out):
                logger.warning(
                    f"Verify missing score. Verify preview (last 200 chars): {repr(verify_out[-200:])}"
                )
                raise ValueError("verify does not end with 'Score: 1' or 'Score: 0'")

            # Capture provider reasoning if available; fall back to JSON 'think'.
            # Use the helper function instead of direct attribute access
            reasoning = _extract_reasoning_from_message(msg)
            if not reasoning:
                reasoning = _extract_reasoning_from_response_dump(resp_dump)
            if not reasoning:
                reasoning = ""  # Default to empty string if no reasoning found

            artifact = {
                "ts": ts,
                "model": model,
                "prompt": prompt,
                # Prefer provider reasoning if present; keep raw JSON 'think' too.
                "reasoning": reasoning,
                "answer": answer_out,
                "verify": verify_out,
                "raw_content": content,
                "raw_response": resp_dump,
                "attempt": attempt + 1,
            }
            return R1Result(ok=True, artifact=artifact)

        except Exception as e:
            # Provide more detailed error information
            error_type = type(e).__name__
            error_msg = str(e)
            if not error_msg:
                error_msg = f"{error_type} occurred"
            last_err = f"{error_type}: {error_msg}"
            
            # Log detailed error information for debugging
            logger.debug(
                f"Attempt {attempt + 1}/{max_tries} failed: {error_type}: {error_msg}"
            )
            
            # Small jittered backoff to play nicely with rate limits and transient failures.
            time.sleep(0.5 + random.random() * 0.75)
            continue

    # Save failure sample if requested
    if save_failures_to and last_err:
        try:
            failure_data = {
                "error": last_err,
                "model": model,
                "max_tries": max_tries,
                "timestamp": _utc_ts(),
                "context": failure_context or {},
                "prompt_preview": prompt[:500] if len(prompt) > 500 else prompt,
            }
            # Append to failure log file
            with open(save_failures_to, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_data, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to save failure sample: {log_err}")
    
    return R1Result(ok=False, error=last_err or "unknown_error")


def _needs_processing(data: Dict[str, Any], filename: str) -> bool:
    # User requirement: if the "R1_approach" key exists at all, skip generation.
    return ("R1_approach" not in data) and ("3obj" in filename)


def process_file(
    path: Path,
    client: OpenAI,
    model: str,
    system_prompt: str,
    max_retries: int,
    save_failures_to: Optional[Path] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Returns (success, filename, error_message).
    """
    try:
        data = _safe_json_load(path)

        # Skip anything that is clearly not valid.
        if not data.get("success", False):
            return False, path.name, "skipped: success != True"
        if not _needs_processing(data, path.name):
            return False, path.name, "skipped: already has R1_approach"
        nl_description = data.get("nl_description", "")
        if not _is_nonempty_str(nl_description):
            return False, path.name, "skipped: missing nl_description"

        ground_truth_json = get_ground_truth_json(data)
        prompt = build_r1_prompt(system_prompt, nl_description, ground_truth_json)
        result = _call_r1_solve(
            client=client,
            model=model,
            prompt=prompt,
            max_tries=max_retries,
            save_failures_to=save_failures_to,
            failure_context={"filename": path.name, "nl_description_length": len(nl_description)},
        )

        if not result.ok or not result.artifact:
            # Log failure with context
            logger.warning(f"Failed to process {path.name}: {result.error}")
            return False, path.name, f"failed: {result.error}"

        data["R1_approach"] = {
            "think": result.artifact["reasoning"],
            "answer": result.artifact["answer"],
            "verify": result.artifact["verify"]
        }
        print("chek chek:")
        print(data["R1_approach"]["think"])
        
        _atomic_write_json(path, data)
        return True, path.name, None

    except Exception as e:
        return False, path.name, f"exception: {str(e)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing json_processed_2 files.",
    )
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--save_failures",
        type=str,
        default=None,
        help="Optional path to save failure samples for analysis (JSONL format)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("API_KEY", "YOUR_API_KEY"))
    base_url = "https://openrouter.ai/api/v1"
    model =  "deepseek/deepseek-r1-0528"

    if not api_key or api_key == "YOUR_API_KEY":
        raise RuntimeError("Set OPENROUTER_API_KEY or API_KEY environment variable (or replace YOUR_API_KEY).")

    system_prompt_path = Path(__file__).resolve().parent / "prompts/general_thinking_prompt.md"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} files in {input_dir}")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Workers: {args.max_workers}")
    print("-" * 80)

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Set up failure logging if requested
    save_failures_to = None
    if args.save_failures:
        save_failures_to = Path(args.save_failures)
        save_failures_to.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Failure samples will be saved to: {save_failures_to}")

    stats = {"success": 0, "failed": 0, "skipped": 0}
    error_types = {}  # Track error types for analysis
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                process_file,
                p,
                client,
                model,
                system_prompt,
                args.max_retries,
                save_failures_to,
            ): p
            for p in json_files
        }

        for fut in as_completed(futures):
            p = futures[fut]
            completed += 1
            try:
                ok, name, err = fut.result()
                if ok:
                    stats["success"] += 1
                    print(f"[{completed}/{len(json_files)}] ✓ {name}")
                else:
                    if err and err.startswith("skipped:"):
                        stats["skipped"] += 1
                        print(f"[{completed}/{len(json_files)}] ⊘ {name} ({err})")
                    else:
                        stats["failed"] += 1
                        # Extract error type for tracking
                        if err:
                            error_type = err.split(":")[0] if ":" in err else "unknown"
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                        print(f"[{completed}/{len(json_files)}] ✗ {name} ({err})")
            except Exception as e:
                stats["failed"] += 1
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
                print(f"[{completed}/{len(json_files)}] ✗ {p.name} (exception: {str(e)})")

    print("-" * 80)
    print("Done")
    print(json.dumps(stats, indent=2))
    if error_types:
        print("\nError type breakdown:")
        for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {err_type}: {count}")
    if save_failures_to and save_failures_to.exists():
        failure_count = sum(1 for _ in open(save_failures_to, "r", encoding="utf-8"))
        print(f"\nFailure samples saved: {failure_count} entries in {save_failures_to}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


