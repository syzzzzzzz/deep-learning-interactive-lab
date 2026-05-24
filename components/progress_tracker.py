"""学习进度、复习队列、学习报告和错题/实验记录。"""

from __future__ import annotations

import base64
import json
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any
from urllib.parse import quote


PROGRESS_STATUSES = ("未学习", "已学习", "已掌握", "去实战", "稍后复习")
COMPLETION_STATUSES = ("已学习", "已掌握", "去实战")
RECORD_TYPES = ("错题", "实验记录", "学习笔记")

_STATE_KEY = "learning_progress"
_PROFILE_STATE_KEY = "learning_profile"
_LOCAL_STORAGE_KEY = "dl_book_progress"
_PROFILE_STORAGE_KEY = "dl_book_profile"
_QUERY_KEY = "_dl_book_progress"
_PROFILE_QUERY_KEY = "_dl_book_profile"


def _st():
    import streamlit as st

    return st


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _components_html(html: str) -> None:
    st = _st()
    try:
        st.components.v1.html(html, height=0)
    except AttributeError:
        import streamlit.components.v1 as components

        components.html(html, height=0)


def normalize_module_key(module_key: str) -> str:
    cleaned = str(module_key).strip().replace("\\", "/")
    parts = cleaned.split("/")
    if len(parts) == 2:
        folder_map = {
            "part1_foundations": "part1",
            "part2_cnn": "part2",
            "part3_rnn": "part3",
            "part4_transformer": "part4",
            "part5_toolbox": "part5",
            "part6_universal_framework": "part6",
            "part7_interview": "part7",
        }
        cleaned = f"{folder_map.get(parts[0], parts[0])}/{parts[1]}"
    try:
        from components.knowledge_graph import get_node

        node = get_node(cleaned)
        return node.name if node else cleaned
    except Exception:
        return cleaned


def _sanitize_progress(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    progress: dict[str, str] = {}
    for module_key, status in value.items():
        if isinstance(module_key, str) and status in PROGRESS_STATUSES:
            progress[normalize_module_key(module_key)] = status
    return progress


def _default_profile() -> dict[str, Any]:
    return {"review_later": {}, "records": []}


def _sanitize_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _default_profile()

    review_later: dict[str, dict[str, str]] = {}
    raw_review = value.get("review_later", {})
    if isinstance(raw_review, dict):
        for module_key, payload in raw_review.items():
            key = normalize_module_key(str(module_key))
            if isinstance(payload, dict):
                review_later[key] = {
                    "reason": str(payload.get("reason", "")).strip()[:240],
                    "priority": str(payload.get("priority", "普通"))[:20],
                    "created_at": str(payload.get("created_at", ""))[:32] or _now(),
                }
            elif payload:
                review_later[key] = {"reason": "", "priority": "普通", "created_at": _now()}

    records: list[dict[str, Any]] = []
    raw_records = value.get("records", [])
    if isinstance(raw_records, list):
        for item in raw_records[-200:]:
            if not isinstance(item, dict):
                continue
            module_key = normalize_module_key(str(item.get("module_key", "")))
            if not module_key:
                continue
            record_type = str(item.get("type", "学习笔记"))
            if record_type not in RECORD_TYPES:
                record_type = "学习笔记"
            records.append(
                {
                    "id": str(item.get("id") or uuid.uuid4().hex),
                    "module_key": module_key,
                    "type": record_type,
                    "title": str(item.get("title", "")).strip()[:120],
                    "note": str(item.get("note", "")).strip()[:2000],
                    "reflection": str(item.get("reflection", "")).strip()[:1200],
                    "created_at": str(item.get("created_at", ""))[:32] or _now(),
                    "linked_nodes": [
                        normalize_module_key(str(key))
                        for key in item.get("linked_nodes", [])
                        if isinstance(key, str)
                    ][:8],
                }
            )

    return {"review_later": review_later, "records": records}


def _json_for_storage(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _query_param_value(st: Any, query_key: str) -> str | None:
    try:
        value = st.query_params.get(query_key)
    except Exception:
        try:
            value = st.experimental_get_query_params().get(query_key)
        except Exception:
            return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _load_from_query(st: Any, query_key: str, sanitizer) -> Any:
    encoded = _query_param_value(st, query_key)
    if not encoded:
        return sanitizer({})
    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return sanitizer(json.loads(raw))
    except Exception:
        return sanitizer({})


def _request_local_storage_restore(storage_key: str, query_key: str, flag_key: str) -> None:
    st = _st()
    if st.session_state.get(flag_key):
        return

    st.session_state[flag_key] = True
    storage = json.dumps(storage_key)
    query = json.dumps(query_key)
    _components_html(
        f"""
        <script>
        (function() {{
            const storageKey = {storage};
            const queryKey = {query};
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


def _clear_restore_query_param(query_key: str) -> None:
    query = json.dumps(query_key)
    _components_html(
        f"""
        <script>
        (function() {{
            const queryKey = {query};
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


def _write_local_storage(storage_key: str, payload: Any, *, merge_existing: bool) -> None:
    storage = json.dumps(storage_key)
    data = json.dumps(_json_for_storage(payload))
    merge = "true" if merge_existing else "false"
    _components_html(
        f"""
        <script>
        (function() {{
            const storageKey = {storage};
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


def _ensure_store(
    state_key: str,
    storage_key: str,
    query_key: str,
    sanitizer,
    default_factory,
) -> Any:
    st = _st()
    local_loaded_key = f"{state_key}_local_loaded"
    restore_requested_key = f"{state_key}_restore_requested"
    if state_key not in st.session_state:
        restored = _load_from_query(st, query_key, sanitizer)
        st.session_state[state_key] = restored if restored else default_factory()
        st.session_state[local_loaded_key] = bool(restored)
        if restored:
            _clear_restore_query_param(query_key)
        else:
            _request_local_storage_restore(storage_key, query_key, restore_requested_key)
    elif not st.session_state.get(local_loaded_key):
        restored = _load_from_query(st, query_key, sanitizer)
        if restored:
            current = sanitizer(st.session_state[state_key])
            if isinstance(restored, dict) and isinstance(current, dict):
                if "records" in restored or "review_later" in restored:
                    merged = _sanitize_profile(
                        {
                            "review_later": {**restored.get("review_later", {}), **current.get("review_later", {})},
                            "records": [*restored.get("records", []), *current.get("records", [])],
                        }
                    )
                else:
                    merged = {**restored, **current}
                st.session_state[state_key] = merged
            st.session_state[local_loaded_key] = True
            _clear_restore_query_param(query_key)
    return sanitizer(st.session_state[state_key])


def _persist_state(state_key: str, storage_key: str, sanitizer) -> None:
    st = _st()
    payload = sanitizer(st.session_state.get(state_key, {}))
    st.session_state[state_key] = payload
    _write_local_storage(storage_key, payload, merge_existing=False)


def _store() -> dict[str, str]:
    return _ensure_store(_STATE_KEY, _LOCAL_STORAGE_KEY, _QUERY_KEY, _sanitize_progress, dict)


def _profile_store() -> dict[str, Any]:
    return _ensure_store(_PROFILE_STATE_KEY, _PROFILE_STORAGE_KEY, _PROFILE_QUERY_KEY, _sanitize_profile, _default_profile)


def get_progress(module_key: str) -> str:
    try:
        return _store().get(normalize_module_key(module_key), "未学习")
    except Exception:
        return "未学习"


def set_progress(module_key: str, status: str) -> None:
    try:
        if status not in PROGRESS_STATUSES:
            status = "未学习"
        store = _store()
        store[normalize_module_key(module_key)] = status
        st = _st()
        st.session_state[_STATE_KEY] = store
        _persist_state(_STATE_KEY, _LOCAL_STORAGE_KEY, _sanitize_progress)
    except Exception as error:
        st = _st()
        st.warning("学习进度暂时无法保存。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def _linked_node_keys(module_key: str) -> list[str]:
    key = normalize_module_key(module_key)
    try:
        from components.knowledge_graph import get_node

        node = get_node(key)
        if node is None:
            return [key]
        linked = [node.name, *node.prerequisites[:2], *node.related[:3], *node.next_steps[:2]]
        return list(dict.fromkeys(linked))
    except Exception:
        return [key]


def mark_review_later(module_key: str, reason: str = "", priority: str = "普通") -> None:
    profile = _profile_store()
    key = normalize_module_key(module_key)
    profile["review_later"][key] = {
        "reason": reason.strip()[:240],
        "priority": priority,
        "created_at": _now(),
    }
    st = _st()
    st.session_state[_PROFILE_STATE_KEY] = _sanitize_profile(profile)
    _persist_state(_PROFILE_STATE_KEY, _PROFILE_STORAGE_KEY, _sanitize_profile)
    set_progress(key, "稍后复习")


def clear_review_later(module_key: str) -> None:
    profile = _profile_store()
    key = normalize_module_key(module_key)
    profile["review_later"].pop(key, None)
    st = _st()
    st.session_state[_PROFILE_STATE_KEY] = _sanitize_profile(profile)
    _persist_state(_PROFILE_STATE_KEY, _PROFILE_STORAGE_KEY, _sanitize_profile)


def add_learning_record(
    module_key: str,
    record_type: str,
    title: str,
    note: str,
    reflection: str = "",
) -> dict[str, Any]:
    profile = _profile_store()
    key = normalize_module_key(module_key)
    if record_type not in RECORD_TYPES:
        record_type = "学习笔记"
    record = {
        "id": uuid.uuid4().hex,
        "module_key": key,
        "type": record_type,
        "title": title.strip()[:120] or f"{record_type} - {key}",
        "note": note.strip()[:2000],
        "reflection": reflection.strip()[:1200],
        "created_at": _now(),
        "linked_nodes": _linked_node_keys(key),
    }
    profile["records"].append(record)
    st = _st()
    st.session_state[_PROFILE_STATE_KEY] = _sanitize_profile(profile)
    _persist_state(_PROFILE_STATE_KEY, _PROFILE_STORAGE_KEY, _sanitize_profile)
    return record


def get_learning_records(module_key: str | None = None) -> list[dict[str, Any]]:
    profile = _profile_store()
    records = list(profile.get("records", []))
    if module_key is not None:
        key = normalize_module_key(module_key)
        records = [record for record in records if record.get("module_key") == key]
    return records


def module_url(module_key: str) -> str:
    key = normalize_module_key(module_key)
    try:
        from components.knowledge_graph import get_node

        node = get_node(key)
        route = node.route if node else key
    except Exception:
        route = key
    return f"/?module={quote(route, safe='')}"


def analyze_weaknesses(
    progress: dict[str, str] | None = None,
    profile: dict[str, Any] | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    progress = _sanitize_progress(progress if progress is not None else _store())
    profile = _sanitize_profile(profile if profile is not None else _profile_store())
    try:
        from components.knowledge_graph import canonical_node_keys, get_node
    except Exception:
        return []

    review_later = set(profile.get("review_later", {}))
    wrong_counts = Counter(
        record["module_key"]
        for record in profile.get("records", [])
        if record.get("type") == "错题"
    )
    experiment_counts = Counter(
        record["module_key"]
        for record in profile.get("records", [])
        if record.get("type") == "实验记录"
    )

    weakness_rows: list[dict[str, Any]] = []
    for key in canonical_node_keys():
        node = get_node(key)
        if node is None:
            continue
        status = progress.get(key, "未学习")
        score = 0
        reasons: list[str] = []
        if status == "稍后复习":
            score += 45
            reasons.append("已加入稍后复习")
        elif status == "未学习":
            score += 12
        elif status == "已学习":
            score += 8
        if key in review_later:
            score += 25
            reasons.append("复习队列中")
        if wrong_counts[key]:
            score += wrong_counts[key] * 30
            reasons.append(f"错题 {wrong_counts[key]} 条")
        if node.difficulty in {"核心", "工程", "前沿"} and status != "已掌握":
            score += 8
            reasons.append(f"{node.difficulty}节点未掌握")
        unmet = [
            prereq
            for prereq in node.prerequisites
            if progress.get(prereq, "未学习") not in COMPLETION_STATUSES
        ]
        if unmet and status != "未学习":
            score += min(18, len(unmet) * 6)
            reasons.append("前置知识未补齐")
        if "实验" in node.tags and experiment_counts[key] == 0 and status in {"已学习", "去实战"}:
            score += 10
            reasons.append("缺少实验记录")
        if score > 0:
            weakness_rows.append(
                {
                    "key": key,
                    "title": node.title,
                    "score": score,
                    "reasons": reasons or ["尚未形成稳定掌握证据"],
                    "action": node.practice_target,
                    "url": module_url(key),
                }
            )

    weakness_rows.sort(key=lambda item: (-item["score"], item["title"]))
    return weakness_rows[:limit]


def recommend_today(progress: dict[str, str] | None = None, profile: dict[str, Any] | None = None) -> dict[str, Any] | None:
    progress = _sanitize_progress(progress if progress is not None else _store())
    profile = _sanitize_profile(profile if profile is not None else _profile_store())
    try:
        from components.knowledge_graph import canonical_node_keys, get_node
    except Exception:
        return None

    review_later = profile.get("review_later", {})
    if review_later:
        priority_order = {"高": 0, "普通": 1, "低": 2}
        candidates = sorted(
            review_later,
            key=lambda key: (priority_order.get(review_later[key].get("priority", "普通"), 1), review_later[key].get("created_at", "")),
        )
        key = candidates[0]
        node = get_node(key)
        if node:
            return {"key": key, "title": node.title, "reason": "你之前把它加入了稍后复习。", "url": module_url(key)}

    weaknesses = analyze_weaknesses(progress, profile, limit=1)
    if weaknesses:
        weak = weaknesses[0]
        if weak["score"] >= 30:
            return {"key": weak["key"], "title": weak["title"], "reason": "弱点分析认为这里最值得补。", "url": weak["url"]}

    keys = canonical_node_keys()
    ready: list[str] = []
    for key in keys:
        node = get_node(key)
        if node is None or progress.get(key, "未学习") in COMPLETION_STATUSES:
            continue
        if all(progress.get(prereq, "未学习") in COMPLETION_STATUSES for prereq in node.prerequisites):
            ready.append(key)
    if not ready:
        ready = [key for key in keys if progress.get(key, "未学习") != "已掌握"] or keys
    if not ready:
        return None
    key = ready[date.today().toordinal() % len(ready)]
    node = get_node(key)
    return {"key": key, "title": node.title, "reason": "根据当前进度和前置知识自动挑选。", "url": module_url(key)}


def build_learning_report(progress: dict[str, str] | None = None, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = _sanitize_progress(progress if progress is not None else _store())
    profile = _sanitize_profile(profile if profile is not None else _profile_store())
    try:
        from components.knowledge_graph import canonical_node_keys, get_node
    except Exception:
        keys = sorted(progress)
        nodes = {}
    else:
        keys = canonical_node_keys()
        nodes = {key: get_node(key) for key in keys}

    counts = Counter(progress.get(key, "未学习") for key in keys)
    completed = sum(counts.get(status, 0) for status in COMPLETION_STATUSES)
    part_counts: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in keys:
        status = progress.get(key, "未学习")
        part_counts[key.split("/", 1)[0]][status] += 1
        node = nodes.get(key)
        difficulty_counts[node.difficulty if node else "未知"][status] += 1
    records = profile.get("records", [])
    record_counts = Counter(record.get("type", "学习笔记") for record in records)
    return {
        "total": len(keys),
        "counts": dict(counts),
        "completed": completed,
        "completion_ratio": completed / max(1, len(keys)),
        "review_count": len(profile.get("review_later", {})),
        "record_count": len(records),
        "record_counts": dict(record_counts),
        "part_counts": {key: dict(value) for key, value in part_counts.items()},
        "difficulty_counts": {key: dict(value) for key, value in difficulty_counts.items()},
        "today": recommend_today(progress, profile),
        "weaknesses": analyze_weaknesses(progress, profile, limit=5),
    }


def render进度标记(module_key: str) -> None:
    """渲染可点击的进度标记按钮。"""

    try:
        st = _st()
        key = normalize_module_key(module_key)
        current = get_progress(key)
        st.markdown(f"**当前进度：{current}**")
        cols = st.columns(len(PROGRESS_STATUSES))
        for index, status in enumerate(PROGRESS_STATUSES):
            label = f"设为{status}" if status != current else f"当前：{status}"
            if cols[index].button(label, key=f"progress-{key}-{status}", width="stretch"):
                set_progress(key, status)
                st.rerun()
    except Exception as error:
        st = _st()
        st.warning("进度标记暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")


def render章节完成标准(module_key: str) -> None:
    try:
        st = _st()
        from components.knowledge_graph import get_node

        node = get_node(normalize_module_key(module_key))
        if node is None:
            return
        st.markdown("**章节完成标准**")
        st.markdown(f"- 掌握标准：{node.mastery_criteria}")
        st.markdown(f"- 实战证明：{node.practice_target}")
        if node.prerequisites:
            missing = [key for key in node.prerequisites if get_progress(key) not in COMPLETION_STATUSES]
            if missing:
                st.warning("前置知识还没完全补齐：" + "、".join(get_node(key).title for key in missing if get_node(key)))
            else:
                st.success("前置知识已满足，可以放心推进本章。")
    except Exception as error:
        st = _st()
        st.warning("章节完成标准暂时无法显示。")
        with st.expander("查看完成标准错误", expanded=False):
            st.code(str(error), language="text")


def _render_record_list(module_key: str | None = None, limit: int = 5) -> None:
    st = _st()
    records = get_learning_records(module_key)
    if not records:
        st.caption("还没有记录。错题、实验现象和一句复盘都可以放在这里。")
        return
    for record in reversed(records[-limit:]):
        st.markdown(f"**{record['type']}｜{record['title']}**")
        st.caption(f"{record['created_at']}｜{record['module_key']}")
        if record.get("note"):
            st.write(record["note"])
        if record.get("reflection"):
            st.caption("复盘：" + record["reflection"])
        linked = [key for key in record.get("linked_nodes", []) if key != record["module_key"]][:4]
        if linked:
            links = " ".join(f"[{key}]({module_url(key)})" for key in linked)
            st.markdown("关联知识：" + links)


def render复习与记录面板(module_key: str) -> None:
    try:
        st = _st()
        key = normalize_module_key(module_key)
        left, right = st.columns([0.42, 0.58])
        with left:
            st.markdown("**稍后复习**")
            priority = st.selectbox("复习优先级", ["普通", "高", "低"], key=f"review-priority-{key}")
            reason = st.text_input("为什么稍后复习", placeholder="例如：公式没看懂 / 调参现象没解释清", key=f"review-reason-{key}")
            c1, c2 = st.columns(2)
            if c1.button("加入稍后复习", key=f"review-add-{key}", width="stretch"):
                mark_review_later(key, reason, priority)
                st.success("已加入稍后复习，并同步到学习报告。")
            if c2.button("移出复习队列", key=f"review-clear-{key}", width="stretch"):
                clear_review_later(key)
                st.success("已移出复习队列。")
        with right:
            st.markdown("**错题 / 实验记录**")
            with st.form(f"record-form-{key}", clear_on_submit=True):
                record_type = st.selectbox("记录类型", list(RECORD_TYPES))
                title = st.text_input("记录标题", placeholder="例如：softmax 温度调大后注意力变平")
                note = st.text_area("现象 / 错因 / 实验配置", height=90)
                reflection = st.text_area("一句复盘", height=70, placeholder="下次遇到同类问题先检查什么？")
                submitted = st.form_submit_button("保存记录", width="stretch")
                if submitted:
                    add_learning_record(key, record_type, title, note, reflection)
                    st.success("记录已保存，并和当前知识图谱节点建立关联。")
        st.markdown("**本章最近记录**")
        _render_record_list(key)
    except Exception as error:
        st = _st()
        st.warning("复习与记录面板暂时无法显示。")
        with st.expander("查看记录面板错误", expanded=False):
            st.code(str(error), language="text")


def render学习操作面板(module_key: str) -> None:
    st = _st()
    st.divider()
    st.subheader("学习进度与复盘")
    render进度标记(module_key)
    render章节完成标准(module_key)
    render复习与记录面板(module_key)


def render今日推荐() -> None:
    st = _st()
    item = recommend_today()
    if item is None:
        st.info("还没有足够数据生成今日推荐。")
        return
    st.markdown(f"**{item['title']}**")
    st.caption(item["reason"])
    st.link_button("打开今日推荐", item["url"], width="stretch")


def render弱点分析() -> None:
    st = _st()
    weaknesses = analyze_weaknesses()
    if not weaknesses:
        st.success("暂时没有明显弱点。继续学习后，错题、复习队列和前置缺口会自动汇总到这里。")
        return
    for item in weaknesses:
        st.markdown(f"**{item['title']}**｜弱点分 {item['score']}")
        st.caption("；".join(item["reasons"]))
        st.write(item["action"])
        st.link_button("去补这一块", item["url"], width="stretch")


def render学习报告() -> None:
    st = _st()
    report = build_learning_report()
    cols = st.columns(4)
    cols[0].metric("总章节", report["total"])
    cols[1].metric("完成章节", report["completed"])
    cols[2].metric("完成率", f"{report['completion_ratio']:.0%}")
    cols[3].metric("复习 / 记录", f"{report['review_count']} / {report['record_count']}")

    st.markdown("**状态分布**")
    rows = [{"状态": status, "数量": report["counts"].get(status, 0)} for status in PROGRESS_STATUSES]
    st.dataframe(rows, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**今日推荐**")
        render今日推荐()
    with right:
        st.markdown("**弱点分析**")
        render弱点分析()

    st.markdown("**最近错题 / 实验记录**")
    _render_record_list(None, limit=8)


def render进度总览() -> None:
    """渲染所有模块的进度概览、今日推荐、学习报告和弱点分析。"""

    try:
        st = _st()
        store = _store()
        graph = {}
        try:
            from components.knowledge_graph import KNOWLEDGE_GRAPH, canonical_node_keys

            graph = KNOWLEDGE_GRAPH
            module_keys = canonical_node_keys()
        except Exception:
            module_keys = sorted(store)

        tabs = st.tabs(["学习报告", "复习队列", "章节表", "导出"])
        with tabs[0]:
            render学习报告()
        with tabs[1]:
            profile = _profile_store()
            review = profile.get("review_later", {})
            if not review:
                st.info("复习队列为空。打开任意章节后，可以把暂时没吃透的内容加入“稍后复习”。")
            for key, payload in review.items():
                node = graph.get(key)
                title = node.title if node else key
                st.markdown(f"**{title}**｜{payload.get('priority', '普通')}")
                st.caption(f"{payload.get('created_at', '')}｜{payload.get('reason', '')}")
                cols = st.columns(2)
                cols[0].link_button("打开复习", module_url(key), width="stretch")
                if cols[1].button("标记已复习", key=f"review-done-{key}", width="stretch"):
                    clear_review_later(key)
                    set_progress(key, "已学习")
                    st.rerun()
        with tabs[2]:
            rows = []
            for key in module_keys:
                node = graph.get(key)
                rows.append(
                    {
                        "模块": node.title if node else key,
                        "进度": store.get(key, "未学习"),
                        "掌握标准": node.mastery_criteria if node else "",
                        "实战目标": node.practice_target if node else "",
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)
        with tabs[3]:
            profile = _profile_store()
            export_payload = {"progress": _sanitize_progress(store), "profile": _sanitize_profile(profile)}
            st.download_button(
                "导出学习档案",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="dl_book_learning_profile.json",
                mime="application/json",
                width="stretch",
            )
            st.caption("导出内容包含进度状态、稍后复习队列、错题、实验记录和学习笔记。")
    except Exception as error:
        st = _st()
        st.warning("学习进度总览暂时无法显示。")
        with st.expander("查看进度错误详情", expanded=False):
            st.code(str(error), language="text")
