"""基于 Streamlit session_state 和 localStorage 的轻量学习进度。"""

from __future__ import annotations

import base64
import json
from collections import Counter
from typing import Any


PROGRESS_STATUSES = ("未学习", "已学习", "已掌握", "去实战")
_STATE_KEY = "learning_progress"
_LOCAL_STORAGE_KEY = "dl_book_progress"
_QUERY_KEY = "_dl_book_progress"
_LOCAL_LOADED_KEY = f"{_STATE_KEY}_local_storage_loaded"
_RESTORE_REQUESTED_KEY = f"{_STATE_KEY}_local_storage_restore_requested"
_LAST_PERSISTED_KEY = f"{_STATE_KEY}_last_persisted_json"
_PENDING_PERSIST_KEY = f"{_STATE_KEY}_pending_persist_json"


def _st():
    import streamlit as st

    return st


def _components_html(html: str) -> None:
    st = _st()
    try:
        st.components.v1.html(html, height=0)
    except AttributeError:
        import streamlit.components.v1 as components

        components.html(html, height=0)


def _sanitize_progress(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    progress: dict[str, str] = {}
    for module_key, status in value.items():
        if isinstance(module_key, str) and status in PROGRESS_STATUSES:
            progress[module_key] = status
    return progress


def _progress_json(progress: dict[str, str]) -> str:
    return json.dumps(_sanitize_progress(progress), ensure_ascii=False, sort_keys=True)


def _query_param_value(st: Any) -> str | None:
    try:
        value = st.query_params.get(_QUERY_KEY)
    except Exception:
        try:
            value = st.experimental_get_query_params().get(_QUERY_KEY)
        except Exception:
            return None

    if isinstance(value, list):
        return value[0] if value else None
    return value


def _load_progress_from_query(st: Any) -> dict[str, str]:
    encoded = _query_param_value(st)
    if not encoded:
        return {}

    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return _sanitize_progress(json.loads(raw))
    except Exception:
        return {}


def _request_local_storage_restore() -> None:
    st = _st()
    if st.session_state.get(_RESTORE_REQUESTED_KEY):
        return

    st.session_state[_RESTORE_REQUESTED_KEY] = True
    storage_key = json.dumps(_LOCAL_STORAGE_KEY)
    query_key = json.dumps(_QUERY_KEY)
    _components_html(
        f"""
        <script>
        (function() {{
            const storageKey = {storage_key};
            const queryKey = {query_key};
            try {{
                const raw = window.localStorage.getItem(storageKey);
                if (!raw) return;

                const url = new URL(window.parent.location.href);
                if (url.searchParams.has(queryKey)) return;

                const bytes = new TextEncoder().encode(raw);
                let binary = "";
                bytes.forEach((byte) => binary += String.fromCharCode(byte));
                const encoded = btoa(binary)
                    .replace(/\\+/g, "-")
                    .replace(/\\//g, "_")
                    .replace(/=+$/g, "");

                url.searchParams.set(queryKey, encoded);
                window.parent.location.replace(url.toString());
            }} catch (error) {{}}
        }})();
        </script>
        """
    )


def _clear_restore_query_param() -> None:
    query_key = json.dumps(_QUERY_KEY)
    _components_html(
        f"""
        <script>
        (function() {{
            const queryKey = {query_key};
            try {{
                const url = new URL(window.parent.location.href);
                if (!url.searchParams.has(queryKey)) return;
                url.searchParams.delete(queryKey);
                window.parent.history.replaceState(null, "", url.toString());
            }} catch (error) {{}}
        }})();
        </script>
        """
    )


def _write_local_storage(progress: dict[str, str], *, merge_existing: bool) -> None:
    storage_key = json.dumps(_LOCAL_STORAGE_KEY)
    data = json.dumps(_progress_json(progress))
    merge = "true" if merge_existing else "false"
    _components_html(
        f"""
        <script>
        (function() {{
            const storageKey = {storage_key};
            const data = {data};
            const shouldMerge = {merge};
            try {{
                let next = JSON.parse(data);
                if (shouldMerge) {{
                    const raw = window.localStorage.getItem(storageKey);
                    const existing = raw ? JSON.parse(raw) : {{}};
                    if (existing && typeof existing === "object" && !Array.isArray(existing)) {{
                        next = Object.assign(existing, next);
                    }}
                }}
                window.localStorage.setItem(storageKey, JSON.stringify(next));
            }} catch (error) {{
                window.localStorage.setItem(storageKey, data);
            }}
        }})();
        </script>
        """
    )


def _flush_pending_persist(st: Any, store: dict[str, str]) -> None:
    if not store:
        return

    data = _progress_json(store)
    pending = st.session_state.get(_PENDING_PERSIST_KEY)
    if pending != data and st.session_state.get(_LAST_PERSISTED_KEY) == data:
        return

    _write_local_storage(store, merge_existing=not st.session_state.get(_LOCAL_LOADED_KEY, False))
    st.session_state[_LAST_PERSISTED_KEY] = data
    st.session_state.pop(_PENDING_PERSIST_KEY, None)


def _store() -> dict[str, str]:
    st = _st()
    if _STATE_KEY not in st.session_state:
        restored = _load_progress_from_query(st)
        st.session_state[_STATE_KEY] = restored
        st.session_state[_LOCAL_LOADED_KEY] = bool(restored)
        if restored:
            _clear_restore_query_param()
        else:
            _request_local_storage_restore()
    elif not st.session_state.get(_LOCAL_LOADED_KEY):
        restored = _load_progress_from_query(st)
        if restored:
            restored.update(_sanitize_progress(st.session_state[_STATE_KEY]))
            st.session_state[_STATE_KEY] = restored
            st.session_state[_LOCAL_LOADED_KEY] = True
            _clear_restore_query_param()

    _flush_pending_persist(st, st.session_state[_STATE_KEY])
    return st.session_state[_STATE_KEY]


def get_progress(module_key: str) -> str:
    try:
        return _store().get(module_key, "未学习")
    except Exception:
        return "未学习"


def set_progress(module_key: str, status: str) -> None:
    try:
        if status not in PROGRESS_STATUSES:
            status = "未学习"
        store = _store()
        store[module_key] = status
        st = _st()
        st.session_state[_PENDING_PERSIST_KEY] = _progress_json(store)
        _write_local_storage(store, merge_existing=not st.session_state.get(_LOCAL_LOADED_KEY, False))
    except Exception as error:
        st = _st()
        st.warning("学习进度暂时无法保存。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def render进度标记(module_key: str) -> None:
    """渲染可点击的进度标记按钮。"""
    try:
        st = _st()
        current = get_progress(module_key)
        st.markdown(f"**当前进度：{current}**")
        cols = st.columns(len(PROGRESS_STATUSES))
        for index, status in enumerate(PROGRESS_STATUSES):
            label = f"设为{status}" if status != current else f"当前：{status}"
            if cols[index].button(label, key=f"progress-{module_key}-{status}", width="stretch"):
                set_progress(module_key, status)
                st.rerun()
    except Exception as error:
        st = _st()
        st.warning("进度标记暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def render进度总览() -> None:
    """渲染所有模块的进度概览。"""
    try:
        st = _st()
        store = _store()
        graph = {}
        try:
            from components.knowledge_graph import KNOWLEDGE_GRAPH

            graph = KNOWLEDGE_GRAPH
            module_keys = list(KNOWLEDGE_GRAPH)
        except Exception:
            module_keys = sorted(store)
        rows = [{"模块": graph[key].title if key in graph else key, "进度": store.get(key, "未学习")} for key in module_keys]
        counts = Counter(row["进度"] for row in rows)
        cols = st.columns(len(PROGRESS_STATUSES))
        for index, status in enumerate(PROGRESS_STATUSES):
            cols[index].metric(status, counts.get(status, 0))

        st.download_button(
            "导出进度",
            data=_progress_json(store),
            file_name="dl_book_progress.json",
            mime="application/json",
        )

        if not module_keys:
            st.info("还没有记录学习进度。打开任意模块后，可以把它标记为已学习、已掌握或去实战。")
            return

        st.dataframe(rows, width="stretch", hide_index=True)
    except Exception as error:
        st = _st()
        st.warning("学习进度总览暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")
