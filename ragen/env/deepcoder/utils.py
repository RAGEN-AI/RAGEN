# Based on code from:
# https://github.com/rllm-org/rllm/tree/main/examples/deepcoder
#
# This file has been modified from the original implementation
# to fit the DeepCoder environment used in this project.
import json
import os
import subprocess
import tempfile
import textwrap
from typing import Optional, Tuple

from datasets import concatenate_datasets, load_dataset

try:
    from rllm.data.dataset import DatasetRegistry
except Exception:
    DatasetRegistry = None

try:
    from rllm.data.utils import fetch_live_code_bench_system_prompt
except Exception:
    def fetch_live_code_bench_system_prompt(problem: str, starter_code: str | None = None) -> str:
        if starter_code:
            return f"{problem}\n\nStarter code:\n{starter_code}"
        return problem


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = concatenate_datasets([
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train"),
    ])
    test_dataset = concatenate_datasets([
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test"),
    ])

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(
            example["problem"],
            starter_code if starter_code else None,
        )

        tests_raw = example["tests"]
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({
                    "input": input_val,
                    "output": output_val,
                    "testtype": "stdin_stdout",
                })
            tests = normalized_tests

        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"deepcoder_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "metadata": json.dumps(metadata),
        }

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)

    if DatasetRegistry is not None:
        train_dataset = DatasetRegistry.register_dataset("deepcoder", train_dataset, "train")
        test_dataset = DatasetRegistry.register_dataset("deepcoder", test_dataset, "test")

    return train_dataset, test_dataset


def _normalize_top_level_code(code: str) -> str:
    return code.strip("\n")


def run_deepcoder_sandbox(
    code: str,
    tests: list,
    metadata: Optional[dict] = None,
    starter_code: str = "",
    timeout_seconds: int = 3,
) -> Tuple[bool, str, int, int, bool]:
    if not tests:
        return False, "No tests available.", 0, 0, False

    func_name = None
    if metadata:
        func_name = metadata.get("func_name")

    header = _normalize_top_level_code(starter_code or "")
    body = _normalize_top_level_code(code or "")

    header_has_class = "class Solution" in header
    body_has_class = "class Solution" in body

    if body_has_class:
        # If the model already provides the class, ignore starter_code to avoid duplication.
        header = ""
    elif header_has_class and body.startswith("def "):
        # If starter_code defines the class and body provides a method, indent body into class.
        indented = []
        for line in body.splitlines():
            indented.append(("    " + line) if line.strip() else "")
        body = "\n".join(indented)

    if header and body:
        user_code = f"{header}\n\n{body}"
    else:
        user_code = header or body

    prepared_tests = []
    for test in tests:
        testtype = test.get("testtype")
        expected = test.get("output")
        if testtype == "functional" and func_name:
            raw_input = test.get("input", [])
            if isinstance(raw_input, str):
                parts = [p for p in raw_input.strip().splitlines() if p.strip()]
                try:
                    args = [eval(p) for p in parts]
                except Exception:
                    args = [raw_input]
            elif isinstance(raw_input, (list, tuple)):
                args = list(raw_input)
            else:
                args = [raw_input]
            prepared_tests.append(
                {
                    "testtype": "functional",
                    "args": args,
                    "expected": expected,
                }
            )
        else:
            prepared_tests.append(
                {
                    "testtype": "stdin_stdout",
                    "input": test.get("input", ""),
                    "expected": expected,
                }
            )

    # Run all tests in a single Python child process to avoid process spawn overhead per test.
    harness = f"""
import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr

USER_CODE = json.loads({json.dumps(json.dumps(user_code, ensure_ascii=False), ensure_ascii=False)})
TESTS = json.loads({json.dumps(json.dumps(prepared_tests, ensure_ascii=False), ensure_ascii=False)})
FUNC_NAME = json.loads({json.dumps(json.dumps(func_name, ensure_ascii=False), ensure_ascii=False)})

def _run_functional(test):
    scope = {{}}
    with io.StringIO() as _buf_out, io.StringIO() as _buf_err:
        with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
            exec(USER_CODE, scope, scope)
    call_target = None
    if "Solution" in scope and FUNC_NAME:
        call_target = getattr(scope["Solution"](), FUNC_NAME)
    elif FUNC_NAME and FUNC_NAME in scope:
        call_target = scope[FUNC_NAME]
    else:
        return False, None, "Function not found"
    try:
        result = call_target(*test.get("args", []))
        return True, result, None
    except Exception as e:
        return False, None, repr(e)

def _run_stdin_stdout(test):
    scope = {{"__name__": "__main__"}}
    stdin_data = str(test.get("input", ""))
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        old_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin_data)
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(USER_CODE, scope, scope)
        return True, out_buf.getvalue(), None
    except SystemExit as e:
        # A program exit ends this test, not the shared test harness.
        if e.code is None or (isinstance(e.code, int) and e.code == 0):
            return True, out_buf.getvalue(), None
        return False, out_buf.getvalue(), repr(e)
    except Exception as e:
        return False, out_buf.getvalue(), repr(e)
    finally:
        sys.stdin = old_stdin

def main():
    passed_tests = 0
    total_tests = len(TESTS)
    runnable = True
    first_failure = None
    events = []

    for idx, test in enumerate(TESTS):
        testtype = test.get("testtype")
        expected = test.get("expected")
        if testtype == "functional":
            ok, result, err = _run_functional(test)
            if not ok:
                runnable = False
                events.append({{"idx": idx, "ok": False}})
                if first_failure is None:
                    first_failure = f"Exception on test {{idx}}: {{err}}"
                continue
            if str(result) != str(expected):
                events.append({{"idx": idx, "ok": False}})
                if first_failure is None:
                    first_failure = f"Wrong answer on test {{idx}}: got {{result}} expected {{expected}}"
                continue
            passed_tests += 1
            events.append({{"idx": idx, "ok": True}})
        else:
            ok, out, err = _run_stdin_stdout(test)
            if not ok:
                runnable = False
                events.append({{"idx": idx, "ok": False}})
                if first_failure is None:
                    first_failure = f"Runtime error on test {{idx}}: {{err}}"
                continue
            got = str(out).strip()
            exp = str(expected).strip()
            if got != exp:
                events.append({{"idx": idx, "ok": False}})
                if first_failure is None:
                    first_failure = f"Wrong answer on test {{idx}}: got {{got}} expected {{exp}}"
                continue
            passed_tests += 1
            events.append({{"idx": idx, "ok": True}})

    is_correct = (passed_tests == total_tests and total_tests > 0)
    if is_correct:
        detail = "All tests passed."
    else:
        summary = f"Passed {{passed_tests}}/{{total_tests}} tests."
        detail = summary if first_failure is None else f"{{summary}} {{first_failure}}"

    print(
        json.dumps(
            {{
                "is_correct": is_correct,
                "detail": detail,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "runnable": runnable,
                "events": events,
            }},
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    main()
"""

    ok, out, err = _exec_python(textwrap.dedent(harness), stdin_input="", timeout_seconds=timeout_seconds)
    if not ok:
        detail = f"Sandbox runtime error: {err or out}"
        return False, detail, 0, len(prepared_tests), False

    try:
        payload = json.loads(out.strip().splitlines()[-1])
    except Exception:
        detail = f"Sandbox malformed output: {out.strip()}"
        return False, detail, 0, len(prepared_tests), False

    if not isinstance(payload, dict):
        detail = f"Sandbox malformed payload type: {type(payload).__name__}"
        return False, detail, 0, len(prepared_tests), False

    events = payload.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            idx = event.get("idx", "?")
            ok = bool(event.get("ok", False))
            print(f"[DeepCoderSandbox] test={idx} ok={ok}")

    return (
        bool(payload.get("is_correct", False)),
        str(payload.get("detail", "Unknown")),
        int(payload.get("passed_tests", 0)),
        int(payload.get("total_tests", len(prepared_tests))),
        bool(payload.get("runnable", False)),
    )


def _exec_python(code: str, stdin_input: str, timeout_seconds: int = 3) -> Tuple[bool, str, str]:
    debug_dir = "/tmp/deepcoder_debug"
    os.makedirs(debug_dir, exist_ok=True)
    fd, script_path = tempfile.mkstemp(prefix="solution_", suffix=".py", dir=debug_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        proc = subprocess.run(
            ["python", script_path],
            input=stdin_input,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass
