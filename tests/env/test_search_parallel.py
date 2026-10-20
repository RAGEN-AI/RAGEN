"""Exercise SearchEnv's resolver and actions through the real MCP HTTP client."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import anyio
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ragen.env.search import SearchEnv, SearchEnvConfig
from ragen.env.search.parallel_client import MCP_URL, USER_AGENT, ParallelRetrievalClient
from ragen.env.search.retrieval_client import MockRetrievalClient, RetrievalClient


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.table({"question": ["What is France's capital?"], "ground_truth": ["Paris"]}), path)
    return str(path)


@pytest.fixture
def wire(monkeypatch):
    """Keep the SDK, initialization, discovery and tool dispatch real."""
    # Load runtime annotations while AsyncClient is still the original class.
    import mcp.client.streamable_http  # noqa: F401
    calls = []
    state = {"mode": "ok", "sse": False}
    original_client = httpx.AsyncClient

    async def handle(request):
        body = json.loads(request.content)
        calls.append((request, body))
        assert str(request.url) == MCP_URL
        assert request.headers["user-agent"] == USER_AGENT
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        method = body["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in ("web_search", "web_fetch")]}
        elif method == "tools/call":
            mode = state["mode"]
            if mode == "timeout":
                await anyio.sleep(10)
            if mode == "connection":
                raise httpx.ConnectError("fixture offline", request=request)
            if mode == "http_error":
                return httpx.Response(503)
            data = {"results": [{"title": "Paris", "url": "https://example.org/paris",
                                 "excerpts": ["Paris is the capital of France."]}]}
            if mode == "empty":
                data = {"results": []}
            elif mode == "malformed":
                data = {"results": "not a list"}
            elif mode == "large":
                data = {"results": [{"url": f"https://example.org/{i}", "excerpts": ["x" * 1000]}
                                    for i in range(5)]}
            result = {"content": [{"type": "text", "text": json.dumps(data)}], "structuredContent": data}
            if mode == "text_only":
                del result["structuredContent"]
            elif mode == "quota":
                result = {"isError": True, "content": [{"type": "text", "text": "Free anonymous quota reached"}]}
        else:
            raise AssertionError(method)
        response = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        if state["sse"]:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="event: message\ndata: " + json.dumps(response) + "\n\n")
        return httpx.Response(200, json=response)

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["timeout"] > 0
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    # Ambient credentials must not turn an anonymous installation into a paid one.
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-ambient-test-key")
    return calls, state


def make_env(data_path, **kwargs):
    env = SearchEnv(SearchEnvConfig(train_path=data_path, retrieval_backend="parallel", **kwargs))
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("sse", [False, True])
def test_search_fetch_finish_and_session_reset(data_path, wire, sse):
    calls, state = wire
    state["sse"] = sse
    env = make_env(data_path)
    assert "fetch[<url>]" in env.render()
    for action in ("search[capital of France]", "fetch[https://example.org/paris]", "search[Paris France]"):
        obs, reward, done, info = env.step(action)
        assert "Paris is the capital of France." in obs
        assert "https://example.org/paris" in obs
        assert reward == 0 and not done
        assert info["action_type"] == action.split("[")[0]
    tool_calls = [b["params"] for _, b in calls if b["method"] == "tools/call"]
    session_id = tool_calls[0]["arguments"]["session_id"]
    assert str(UUID(session_id)) == session_id
    assert {x["arguments"]["session_id"] for x in tool_calls} == {session_id}
    assert tool_calls[0]["arguments"]["search_queries"] == ["capital of France"]
    assert tool_calls[1]["arguments"]["urls"] == ["https://example.org/paris"]
    assert {b["method"] for _, b in calls} >= {"initialize", "tools/list", "tools/call"}
    _, reward, done, info = env.step("finish[Paris]")
    assert reward == 1 and done and info["success"]
    env.reset(seed=0)
    env.step("search[capital of France]")
    assert [b for _, b in calls if b["method"] == "tools/call"][-1]["params"]["arguments"]["session_id"] != session_id


def test_interleaved_and_concurrent_episodes(data_path, wire):
    calls, _ = wire
    a, b = make_env(data_path), make_env(data_path)
    # Even a shared adapter has no mutable conversation identity.
    b.client = a.client
    a.step("search[first question]")
    b.step("search[second question]")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda env: env.step("fetch[https://example.org/paris]"), (a, b)))
    tool_calls = [body["params"]["arguments"] for _, body in calls if body["method"] == "tools/call"]
    assert a._session_id != b._session_id
    assert [x["session_id"] for x in tool_calls].count(a._session_id) == 2
    assert [x["session_id"] for x in tool_calls].count(b._session_id) == 2


@pytest.mark.parametrize("mode,message", [
    ("quota", "Free anonymous quota reached"), ("connection", "Parallel retrieval failed"),
    ("http_error", "Parallel retrieval failed"), ("malformed", "invalid response"),
    ("empty", "No relevant documents"), ("text_only", "Paris is the capital"),
])
def test_errors_and_retry_keep_episode(data_path, wire, mode, message):
    calls, state = wire
    env = make_env(data_path)
    state["mode"] = mode
    obs, _, _, _ = env.step("search[capital of France]")
    assert message in obs
    state["mode"] = "ok"
    assert "Paris is the capital" in env.step("search[capital of France]")[0]
    ids = [b["params"]["arguments"]["session_id"] for _, b in calls if b["method"] == "tools/call"]
    assert ids == [env._session_id, env._session_id]


def test_timeout_and_cancellation(data_path, wire, monkeypatch):
    _, state = wire
    env = make_env(data_path, retrieval_timeout=0.2)
    state["mode"] = "timeout"
    obs = env.step("search[capital of France]")[0]
    assert "timed out" in obs
    async def cancel(*args):
        raise asyncio.CancelledError()
    monkeypatch.setattr(env.client, "_call_tool", cancel)
    with pytest.raises(asyncio.CancelledError):
        env.step("search[capital of France]")


@pytest.mark.parametrize("url", ["file:///etc/hosts", "https://user:password@example.org", "https://[invalid", "javascript:alert(1)"])
def test_invalid_urls_do_not_dispatch(data_path, wire, url):
    calls, _ = wire
    env = make_env(data_path)
    assert "HTTP or HTTPS URL" in env.step(f"fetch[{url}]")[0]
    assert calls == []


def test_result_limits_and_max_steps(data_path, wire):
    _, state = wire
    state["mode"] = "large"
    env = make_env(data_path, max_search_results=1, max_total_chars=80, max_steps=1)
    obs, reward, done, _ = env.step("search[capital of France]")
    result = obs.split("\n", 1)[1].split("\n\nMaximum")[0]
    assert len(result) <= 80
    assert "https://example.org/0" in result and "https://example.org/1" not in result
    assert done and reward == 0


def test_default_explicit_local_and_mock(data_path, monkeypatch, wire):
    calls, _ = wire
    local_calls = []
    monkeypatch.setattr(RetrievalClient, "_test_connection", lambda self: None)
    def search(self, query, top_k):
        local_calls.append((self.server_url, query, top_k))
        return "local corpus result"
    monkeypatch.setattr(RetrievalClient, "search", search)
    for server in ("http://127.0.0.1:8000", "http://my-retriever:9000"):
        env = SearchEnv(SearchEnvConfig(train_path=data_path, retrieval_server_url=server))
        env.reset(seed=0)
        assert type(env.client) is RetrievalClient
        assert "fetch[" not in env.render()
        assert "local corpus result" in env.step("search[fixed corpus]")[0]
        assert local_calls[-1][0] == server
    env = make_env(data_path, mock_mode=True)
    assert type(env.client) is MockRetrievalClient
    assert "Mock search result" in env.step("search[test]")[0]
    assert "fetch[" not in env.render()
    assert calls == []
    with pytest.raises(ValueError, match="Unknown retrieval_backend"):
        SearchEnv(SearchEnvConfig(train_path=data_path, retrieval_backend="unknown"))


def test_reset_required_and_invalid_limits(data_path, wire):
    env = SearchEnv(SearchEnvConfig(train_path=data_path, retrieval_backend="parallel"))
    with pytest.raises(RuntimeError, match="reset"):
        env.step("search[test]")
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ParallelRetrievalClient(timeout=timeout)


def test_registered_environment_manager_consumes_results(data_path, wire):
    from omegaconf import OmegaConf
    from ragen.env import REGISTERED_ENVS
    from ragen.llm_agent.es_manager import EnvStateManager

    assert REGISTERED_ENVS["search"] is SearchEnv
    config = OmegaConf.create({
        "es_manager": {
            "format_penalty": -1,
            "val": {"env_groups": 1, "group_size": 2,
                    "env_configs": {"tags": ["WebSearch"], "n_groups": [1]}},
        },
        "custom_envs": {"WebSearch": {
            "env_type": "search", "max_actions_per_traj": 5,
            "parallel_friendly": True, "max_workers": 2,
            "env_config": {"train_path": data_path, "retrieval_backend": "parallel"},
        }},
    })
    manager = EnvStateManager(config, mode="val")
    try:
        manager.reset(seed=0)
        # Scripted actions, with no model inference: verify runtime dispatcher,
        # concurrent environment workers and returned rollout history.
        for action in ("search[capital of France]", "fetch[https://example.org/paris]"):
            outputs = manager.step([
                {"env_id": i, "actions": [action], "llm_response": action, "llm_raw_response": action}
                for i in range(2)
            ])
            assert len(outputs) == 2
            assert all("Paris is the capital" in out["history"][-1]["state"] for out in outputs)
        manager.step([
            {"env_id": i, "actions": ["finish[Paris]"], "llm_response": "Paris", "llm_raw_response": "Paris"}
            for i in range(2)
        ])
        assert all(entry["status"].rewards == [0, 0, 1] for entry in manager.envs)
        assert all(entry["status"].terminated and not entry["status"].truncated for entry in manager.envs)
    finally:
        manager._shutdown_executors()
