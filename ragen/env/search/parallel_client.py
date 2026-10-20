"""Anonymous Parallel Search MCP adapter for synchronous SearchEnv actions."""

import json
import math
from datetime import timedelta
from urllib.parse import urlsplit


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "RAGEN/0.1 (+https://github.com/mll-lab-nu/RAGEN)"


class ParallelRetrievalClient:
    """Use Streamable HTTP without reading API keys or saved credentials.

    The caller owns the episode's session_id. Connections are short lived so
    concurrent environment workers do not share mutable transport state.
    """

    def __init__(self, timeout=30.0, max_results=5, max_total_chars=4000):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("retrieval_timeout must be finite and positive")
        if max_results <= 0 or max_total_chars <= 0:
            raise ValueError("Search result limits must be positive")
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError("Parallel retrieval requires pip install 'ragen[search]'") from exc
        self.timeout = timeout
        self.max_results = max_results
        self.max_total_chars = max_total_chars

    def search(self, query, *, session_id, top_k=None):
        if not query.strip():
            return "Search query must not be empty."
        return self._execute("web_search", {
            "objective": query,
            "search_queries": [query],
            "session_id": session_id,
        }, min(top_k or self.max_results, self.max_results))

    def fetch(self, url, *, session_id):
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid:
            return "Fetch requires an HTTP or HTTPS URL without credentials."
        return self._execute("web_fetch", {
            "urls": [url],
            "session_id": session_id,
        }, 1)

    def _execute(self, name, arguments, limit):
        import anyio

        try:
            result = anyio.run(self._call_tool, name, arguments)
        except TimeoutError:
            return f"Parallel request timed out after {self.timeout} seconds."
        except Exception:
            # Never fall back to a different retrieval regime or expose transport
            # internals in the model observation. Cancellation remains BaseException.
            return "Parallel retrieval failed. Check the connection and service availability."
        return self._format_result(result, limit)

    async def _call_tool(self, name, arguments):
        import anyio
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import Implementation

        # Bound connection, handshake, tool execution and cleanup together.
        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                follow_redirects=False,
            ) as http_client:
                async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write, _):
                    async with ClientSession(
                        read, write, read_timeout_seconds=timedelta(seconds=self.timeout),
                        client_info=Implementation(name="RAGEN", version="0.1"),
                    ) as session:
                        await session.initialize()
                        return await session.call_tool(name, arguments)

    def _format_result(self, result, limit):
        text = "\n".join(block.text for block in result.content if block.type == "text")
        if result.isError:
            return ("Parallel retrieval error: " + text)[:self.max_total_chars]
        data = result.structuredContent
        if not isinstance(data, dict):
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                return "Parallel retrieval returned an invalid response."
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return "Parallel retrieval returned an invalid response."
        documents = []
        for item in data["results"][:limit]:
            if not isinstance(item, dict):
                return "Parallel retrieval returned an invalid response."
            excerpts = item.get("excerpts") or []
            if not isinstance(excerpts, list) or any(not isinstance(x, str) for x in excerpts):
                return "Parallel retrieval returned an invalid response."
            documents.append("\n".join([
                str(item.get("title") or "Untitled"),
                str(item.get("url") or ""),
                "\n".join(excerpts),
            ]))
        if data.get("errors"):
            documents.append("Some pages could not be retrieved.")
        output = "\n\n".join(documents) or "No relevant documents found for the query."
        return output[:self.max_total_chars]
