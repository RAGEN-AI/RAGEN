import pytest

from ragen.env.deepcoder.config import DeepCoderEnvConfig
from ragen.env.deepcoder.env import DeepCoderEnv
from ragen.env.deepcoder.utils import run_deepcoder_sandbox


@pytest.fixture
def env(monkeypatch):
    # Keep dataset downloads out of these reward tests; execution uses real children.
    monkeypatch.setattr(DeepCoderEnv, "_load_data", lambda self: None)
    env = DeepCoderEnv(DeepCoderEnvConfig(max_steps=2))
    env.current_prompt = "Read an integer and print twice its value."
    env.current_tests = [
        {"input": str(value), "output": str(2 * value), "testtype": "stdin_stdout"}
        for value in (1, 2, 3)
    ]
    return env


@pytest.mark.parametrize(
    "exit_statement", ["sys.exit()", "sys.exit(0)", "raise SystemExit(None)"]
)
def test_successful_program_exit_keeps_output_and_runs_remaining_tests(
    env, exit_statement
):
    code = (
        f"import sys\nprint(2 * int(input()))\n{exit_statement}\nprint('unreachable')"
    )

    observation, reward, done, info = env.step(code)

    assert reward == 1.0
    assert done and info["success"] and info["runnable"]
    assert info["passed_tests"] == info["total_tests"] == 3
    assert observation == "Correct!"
    assert env.compute_reward(code) == 1.0


@pytest.mark.parametrize(
    "failure", ["sys.exit(1)", "sys.exit('failed')", "raise ValueError('failed')"]
)
def test_failed_program_keeps_partial_credit_and_runs_later_tests(env, failure):
    code = (
        "import sys\nvalue = int(input())\nprint(2 * value)\n"
        f"if value == 2:\n    {failure}"
    )

    observation, reward, done, info = env.step(code)

    assert reward == pytest.approx(2 / 3)
    assert not done and not info["success"] and not info["runnable"]
    assert info["passed_tests"] == 2 and info["total_tests"] == 3
    assert observation == "Incorrect."
    assert "Runtime error on test 1" in info["detail"]


def test_successful_exit_still_checks_the_answer(env):
    _, reward, done, info = env.step("import sys\nprint(0)\nsys.exit(0)")

    assert reward == 0.0
    assert not done and not info["success"]
    assert info["runnable"]
    assert "Wrong answer on test 0" in info["detail"]


def test_functional_scoring_is_unchanged():
    result = run_deepcoder_sandbox(
        "def double(value):\n    return 2 * value",
        [{"input": [3], "output": 6, "testtype": "functional"}],
        metadata={"func_name": "double"},
    )
    assert result == (True, "All tests passed.", 1, 1, True)
