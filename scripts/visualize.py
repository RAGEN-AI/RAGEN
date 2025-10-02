#!/usr/bin/env python3
"""
Local rollout visualizer.

Usage:
    python scripts/visualize.py --rollout_path results/ [--host 127.0.0.1] [--port 8000]

The script launches a small HTTP server that lets you inspect .pkl files
(containing verl.DataProto dumps) inside the rollout path. Open the printed
URL in a browser to explore directories, select a file, and view its
meta information and non-tensor batches entry by entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

from verl import DataProto

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a local rollout visualizer")
    parser.add_argument("--rollout_path", required=True, help="Directory containing rollout .pkl files")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not attempt to open a browser automatically")
    return parser.parse_args()


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path {path} escapes the rollout root {root}") from exc
    return resolved


def numpy_summary(array: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(array)
    summary: Dict[str, Any] = {
        "__type__": "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "size": int(array.size),
    }
    preview_limit = 32
    flat = array.reshape(-1)
    preview = flat[:preview_limit].tolist()
    summary["preview"] = preview
    summary["preview_count"] = len(preview)
    if array.size <= preview_limit and array.size <= 10_000:
        summary["values"] = array.tolist()
    return summary


def serialize_for_view(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()

    if isinstance(value, dict):
        return {str(key): serialize_for_view(val, depth + 1) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [serialize_for_view(val, depth + 1) for val in value]

    if isinstance(value, np.ndarray):
        return numpy_summary(value)

    return repr(value)


def build_tree(root: Path) -> Dict[str, Any]:
    root_node: Dict[str, Any] = {"name": root.name, "path": "", "type": "dir", "children": []}
    nodes: Dict[str, Dict[str, Any]] = {"": root_node}

    for file_path in sorted(root.rglob("*.pkl")):
        rel_path = file_path.relative_to(root)
        rel_path_posix = rel_path.as_posix()
        parts = rel_path.parts
        if not parts:
            continue

        cumulative = []
        for part in parts[:-1]:
            cumulative.append(part)
            current_key = "/".join(cumulative)
            parent_key = "/".join(cumulative[:-1]) if len(cumulative) > 1 else ""
            if current_key not in nodes:
                node = {"name": part, "path": current_key, "type": "dir", "children": []}
                nodes[current_key] = node
                nodes[parent_key]["children"].append(node)
        file_parent_key = "/".join(parts[:-1]) if len(parts) > 1 else ""
        file_node = {"name": parts[-1], "path": rel_path_posix, "type": "file"}
        nodes[file_parent_key]["children"].append(file_node)

    def sort_children(node: Dict[str, Any]) -> None:
        children = node.get("children")
        if not children:
            return
        children.sort(key=lambda item: (item.get("type") != "dir", item.get("name", "")))
        for child in children:
            if child.get("type") == "dir":
                sort_children(child)

    sort_children(root_node)
    return root_node


def data_proto_to_payload(file_path: Path) -> Dict[str, Any]:
    data = DataProto.load_from_disk(str(file_path))
    length = len(data)
    items: List[Dict[str, Any]] = []
    for idx in range(length):
        try:
            item = data[idx]
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to read item %s from %s: %s", idx, file_path, exc)
            continue
        items.append(
            {
                "index": idx,
                "meta_info": serialize_for_view(item.meta_info),
                "non_tensor_batch": serialize_for_view(item.non_tensor_batch),
            }
        )

    return {
        "path": str(file_path),
        "length": length,
        "meta_info": serialize_for_view(data.meta_info),
        "items": items,
    }


class RolloutExplorer:
    def __init__(self, root: Path):
        self.root = root
        self._tree_cache: Dict[str, Any] | None = None
        self._lock = threading.Lock()

    def tree(self) -> Dict[str, Any]:
        with self._lock:
            if self._tree_cache is None:
                self._tree_cache = build_tree(self.root)
            return self._tree_cache

    @lru_cache(maxsize=32)
    def load_file(self, relative_path: str) -> Dict[str, Any]:
        normalized_path = Path(relative_path)
        target = ensure_within(self.root / normalized_path, self.root)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File {relative_path} not found under {self.root}")
        payload = data_proto_to_payload(target)
        payload["relative_path"] = target.relative_to(self.root).as_posix()
        return payload


HTML_PAGE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Rollout Visualizer</title>
    <style>
        :root {
            color-scheme: light dark;
            --bg: #f7f7fb;
            --panel: #ffffffcc;
            --accent: #4a6cff;
            --accent-soft: #e6ebff;
            --text: #1d1d25;
            --border: #d9d9e3;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        header {
            padding: 14px 24px;
            background: linear-gradient(135deg, var(--accent), #7f9bff);
            color: white;
            font-weight: 600;
            letter-spacing: 0.4px;
        }
        #layout {
            display: flex;
            height: calc(100vh - 56px);
        }
        #sidebar {
            width: 28%;
            max-width: 360px;
            min-width: 240px;
            border-right: 1px solid var(--border);
            background: var(--panel);
            padding: 12px 16px;
            overflow-y: auto;
        }
        #content {
            flex: 1;
            overflow-y: auto;
            padding: 20px 28px;
        }
        .tree-node {
            margin-left: 12px;
        }
        .tree-toggle {
            cursor: pointer;
            user-select: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 6px;
            border-radius: 6px;
        }
        .tree-toggle:hover {
            background: var(--accent-soft);
        }
        .file-entry {
            cursor: pointer;
            display: block;
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 6px;
        }
        .file-entry:hover,
        .file-entry.active {
            background: var(--accent-soft);
            color: var(--accent);
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 16px rgba(76, 96, 255, 0.05);
        }
        .section-title {
            font-weight: 600;
            margin-bottom: 12px;
            font-size: 18px;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        .section-subtitle {
            font-weight: 600;
            margin: 18px 0 10px;
            color: var(--accent);
            font-size: 16px;
        }
        .kv-block {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            background: #fff;
        }
        .kv-header {
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--accent);
        }
        .kv-body {
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        details {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background: #fff;
        }
        details[open] {
            border-color: var(--accent);
            box-shadow: 0 4px 12px rgba(74, 108, 255, 0.08);
        }
        summary {
            cursor: pointer;
            font-weight: 600;
            color: var(--accent);
        }
        .message-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 6px 0;
            background: #fbfbff;
        }
        .message-meta {
            font-size: 13px;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        .message-content {
            white-space: pre-wrap;
            font-family: "Fira Code", "Consolas", monospace;
            font-size: 14px;
        }
        .messages-section {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            background: #f3f5ff;
            margin-bottom: 14px;
            box-shadow: inset 0 0 0 1px rgba(74, 108, 255, 0.05);
        }
        .messages-section .section-subtitle {
            margin-top: 0;
            color: var(--accent);
        }
        .placeholder {
            opacity: 0.6;
            font-style: italic;
        }
    </style>
</head>
<body>
    <header>Rollout Visualizer</header>
    <div id="layout">
        <aside id="sidebar">
            <div id="tree"></div>
        </aside>
        <main id="content">
            <div class="panel placeholder">Select a file to inspect its rollout details.</div>
        </main>
    </div>
    <script>
        let activePath = null;

        async function fetchJSON(url) {
            const res = await fetch(url);
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || res.statusText);
            }
            return res.json();
        }

        function createElement(tag, options = {}) {
            const el = document.createElement(tag);
            if (options.className) el.className = options.className;
            if (options.text) el.textContent = options.text;
            if (options.html) el.innerHTML = options.html;
            return el;
        }

        function renderTree(node, container) {
            const wrapper = createElement('div', { className: 'tree-node' });
            const hasChildren = Array.isArray(node.children) && node.children.length > 0;

            if (node.type === 'dir') {
                const summary = createElement('div', { className: 'tree-toggle' });
                const icon = createElement('span', { text: hasChildren ? '▸' : '•' });
                icon.dataset.state = 'collapsed';
                const label = createElement('span', { text: node.name || '(root)' });
                summary.append(icon, label);
                wrapper.appendChild(summary);
                const childrenContainer = createElement('div');
                childrenContainer.style.display = 'none';
                if (node.path === '') {
                    childrenContainer.style.display = 'block';
                    icon.textContent = '▾';
                }
                summary.addEventListener('click', () => {
                    if (!hasChildren) return;
                    if (childrenContainer.style.display === 'none') {
                        childrenContainer.style.display = 'block';
                        icon.textContent = '▾';
                    } else {
                        childrenContainer.style.display = 'none';
                        icon.textContent = '▸';
                    }
                });
                wrapper.appendChild(childrenContainer);
                node.children.forEach(child => renderTree(child, childrenContainer));
            } else if (node.type === 'file') {
                const entry = createElement('div', { className: 'file-entry', text: node.name });
                entry.dataset.path = node.path;
                entry.addEventListener('click', () => loadFile(node.path, entry));
                wrapper.appendChild(entry);
            }
            container.appendChild(wrapper);
        }

        function renderKeyValue(container, key, value) {
            const block = createElement('div', { className: 'kv-block' });
            block.appendChild(createElement('div', { className: 'kv-header', text: key }));
            const body = createElement('div', { className: 'kv-body' });
            body.appendChild(renderValue(value, key));
            block.appendChild(body);
            container.appendChild(block);
        }

        function renderMessages(messages) {
            const wrapper = createElement('div');
            messages.forEach((msg, idx) => {
                const card = createElement('div', { className: 'message-card' });
                const role = msg.role || msg.author || `Message ${idx}`;
                const meta = createElement('div', { className: 'message-meta', text: `${role}` });
                if (msg.timestamp) {
                    meta.textContent += ` · ${msg.timestamp}`;
                }
                const contentContainer = createElement('div', { className: 'message-content' });
                let content = msg.content;
                if (Array.isArray(content)) {
                    content = content.map(part => typeof part === 'string' ? part : JSON.stringify(part, null, 2)).join('\\n');
                }
                contentContainer.textContent = content ?? '';
                card.append(meta, contentContainer);
                wrapper.appendChild(card);
            });
            return wrapper;
        }

        function renderNdArray(info) {
            const wrapper = createElement('div');
            const summary = `dtype=${info.dtype} · shape=[${info.shape.join(', ')}] · size=${info.size}`;
            wrapper.appendChild(createElement('div', { text: summary }));
            if (info.preview && info.preview.length) {
                const preview = createElement('pre');
                preview.textContent = JSON.stringify(info.preview, null, 2);
                wrapper.appendChild(preview);
            }
            if (info.values) {
                const details = document.createElement('details');
                details.appendChild(createElement('summary', { text: 'Show full values' }));
                const pre = createElement('pre');
                pre.textContent = JSON.stringify(info.values, null, 2);
                details.appendChild(pre);
                wrapper.appendChild(details);
            }
            return wrapper;
        }

        function renderValue(value, key = '') {
            if (value === null || typeof value === 'undefined') {
                return createElement('span', { text: '—' });
            }
            if (typeof value !== 'object') {
                return createElement('span', { text: String(value) });
            }
            if (Array.isArray(value)) {
                if (key === 'messages') {
                    return renderMessages(value.map(item => typeof item === 'object' ? item : { content: String(item) }));
                }
                const details = document.createElement('details');
                details.appendChild(createElement('summary', { text: `List [${value.length}]` }));
                value.forEach((item, idx) => {
                    const line = createElement('div');
                    line.appendChild(createElement('strong', { text: `#${idx}` }));
                    line.appendChild(createElement('div', { className: 'kv-body' }));
                    line.lastChild.appendChild(renderValue(item));
                    details.appendChild(line);
                });
                return details;
            }
            if (value.__type__ === 'ndarray') {
                return renderNdArray(value);
            }
            const entries = Object.entries(value);
            const container = createElement('div');
            entries.forEach(([childKey, childValue]) => {
                const block = createElement('div');
                block.appendChild(createElement('strong', { text: childKey }));
                const inner = createElement('div', { className: 'kv-body' });
                inner.appendChild(renderValue(childValue, childKey));
                block.appendChild(inner);
                container.appendChild(block);
            });
            return container;
        }

        function markActive(entry) {
            document.querySelectorAll('.file-entry.active').forEach(el => el.classList.remove('active'));
            entry.classList.add('active');
        }

        async function loadFile(path, entryEl) {
            try {
                activePath = path;
                markActive(entryEl);
                const data = await fetchJSON(`/api/file?path=${encodeURIComponent(path)}`);
                renderContent(data);
            } catch (error) {
                console.error(error);
                const panel = createElement('div', { className: 'panel' });
                panel.appendChild(createElement('h2', { text: 'Failed to load file' }));
                panel.appendChild(createElement('pre', { text: error.message }));
                const content = document.getElementById('content');
                content.innerHTML = '';
                content.appendChild(panel);
            }
        }

        function renderContent(data) {
            const content = document.getElementById('content');
            content.innerHTML = '';
            const panel = createElement('div', { className: 'panel' });
            const title = createElement('div', { className: 'section-title', text: data.relative_path || data.path });
            panel.appendChild(title);
            panel.appendChild(createElement('div', { text: `Entries: ${data.length}` }));

            const metaSection = createElement('div', { className: 'section-title', text: 'Meta Info (global)' });
            panel.appendChild(metaSection);
            const metaGrid = createElement('div', { className: 'meta-grid' });
            Object.entries(data.meta_info || {}).forEach(([key, value]) => {
                renderKeyValue(metaGrid, key, value);
            });
            if (!Object.keys(data.meta_info || {}).length) {
                metaGrid.appendChild(createElement('div', { className: 'placeholder', text: 'No meta info available.' }));
            }
            panel.appendChild(metaGrid);

            const itemsSection = createElement('div', { className: 'section-title', text: 'Entries' });
            panel.appendChild(itemsSection);
            if (!data.items.length) {
                panel.appendChild(createElement('div', { className: 'placeholder', text: 'No entries in this DataProto.' }));
            }
            data.items.forEach(item => {
                const details = document.createElement('details');
                const summary = createElement('summary', { text: `Item #${item.index}` });
                details.appendChild(summary);

                const metaBlock = createElement('div', { className: 'meta-grid' });
                Object.entries(item.meta_info || {}).forEach(([key, value]) => {
                    renderKeyValue(metaBlock, key, value);
                });
                if (!Object.keys(item.meta_info || {}).length) {
                    metaBlock.appendChild(createElement('div', { className: 'placeholder', text: 'No item-level meta info.' }));
                }
                details.appendChild(metaBlock);

                const nonTensorEntries = Object.entries(item.non_tensor_batch || {});
                let messagesHandled = false;
                if (nonTensorEntries.length) {
                    nonTensorEntries.forEach(([key, value]) => {
                        if (key === 'messages_list') {
                            const messagesSection = createElement('div', { className: 'messages-section' });
                            messagesSection.appendChild(createElement('div', { className: 'section-subtitle', text: 'Messages' }));
                            messagesSection.appendChild(renderValue(value, key));
                            details.appendChild(messagesSection);
                            messagesHandled = true;
                        }
                    });
                    const others = nonTensorEntries.filter(([key]) => key !== 'messages_list');
                    if (others.length) {
                        const ntBlock = createElement('div', { className: 'meta-grid' });
                        others.forEach(([key, value]) => {
                            renderKeyValue(ntBlock, key, value);
                        });
                        details.appendChild(ntBlock);
                    }
                    if (!messagesHandled && !others.length) {
                        details.appendChild(createElement('div', { className: 'placeholder', text: 'No non-tensor batch data.' }));
                    }
                } else {
                    details.appendChild(createElement('div', { className: 'placeholder', text: 'No non-tensor batch data.' }));
                }

                panel.appendChild(details);
            });

            content.appendChild(panel);
        }

        async function init() {
            try {
                const treeData = await fetchJSON('/api/tree');
                const treeRoot = document.getElementById('tree');
                treeRoot.innerHTML = '';
                renderTree(treeData, treeRoot);
            } catch (error) {
                const treeRoot = document.getElementById('tree');
                treeRoot.textContent = 'Failed to load file tree.';
                console.error(error);
            }
        }

        init();
    </script>
</body>
</html>
"""


class VisualizerHandler(BaseHTTPRequestHandler):
    explorer: RolloutExplorer

    def do_GET(self) -> None:  # noqa: N802 - http.server signature
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_html(HTML_PAGE)
            return
        if parsed.path == "/api/tree":
            payload = VisualizerHandler.explorer.tree()
            self.respond_json(payload)
            return
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            relative = query.get("path", [None])[0]
            if not relative:
                self.respond_json({"error": "Missing path query parameter"}, status=400)
                return
            try:
                payload = VisualizerHandler.explorer.load_file(relative)
            except FileNotFoundError:
                self.respond_json({"error": "File not found"}, status=404)
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to load %s", relative)
                self.respond_json({"error": str(exc)}, status=500)
                return
            self.respond_json(payload)
            return

        self.respond_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - inherited name
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def respond_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    root = Path(args.rollout_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Rollout path {root} does not exist or is not a directory")

    explorer = RolloutExplorer(root)
    VisualizerHandler.explorer = explorer

    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)

    address = f"http://{args.host}:{args.port}/"
    print(f"Serving rollout visualizer for {root} at {address}")
    if not args.no_browser:
        try:
            webbrowser.open(address)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.info("Could not open browser automatically: %s", exc)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
