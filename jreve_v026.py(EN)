import math
import json
import re
import os
import time
import logging
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
from datetime import datetime, timedelta
import anthropic
import openai

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("jreve.log"),
        logging.StreamHandler()
    ]
)

# ── API Clients ───────────────────────────────────────────────

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client    = openai.OpenAI(api_key=os.environ.get("CHATGPT_API_KEY"))
deepseek_client  = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── Constants ─────────────────────────────────────────────────

EMAIL_ADDRESS  = os.environ.get("KNIGHT_EMAIL")
EMAIL_PASSWORD = os.environ.get("KNIGHT_EMAIL_PASSWORD")

TASKS_FILE     = "tasks.json"
PROFILE_FILE   = "user_profiles.json"
CONV_DIR       = "conversations"
CACHE_DIR      = "cache"
STATE_DIR      = "state"
DECISIONS_DIR  = os.path.join(STATE_DIR, "decisions")

RECENT_TURNS   = 3
CONF_THRESHOLD = 0.5
USER_KEY       = "local_user"

# ── Identity drift filter patterns ───────────────────────────

IDENTITY_DRIFT_PATTERNS = [
    "我本质上是Claude", "我是Claude", "我基于",
    "底层模型", "state drift", "I am Claude",
    "I'm Claude", "built on", "powered by Anthropic"
]

# ── Values & System Prompts ───────────────────────────────────

JREVE_VALUES = """
Jreve的核心价值观：
1. 用户真实利益优先——不是让用户感觉良好，而是让用户真正接近目标。
2. 忠于用户——Jreve只服务于它的用户，用户的目标、隐私、数据永远不会被用于其他目的。
3. 诚实——张力值是多少就是多少，不美化结果。不确定的时候说不确定。
4. 资源诚实——清楚每次决策的成本，不在低价值任务上浪费用户的时间和token。
5. 主动但有边界——在用户需要的时候主动介入，但不在没有授权的情况下执行不可逆操作。
"""

# 对话类调用：价值观，不含身份（身份由 identity payload 单独注入）
JREVE_SYSTEM = f"""你会根据用户需求调度最合适的模型，帮助用户追踪长期目标、计算差距、制定优先级。
除非用户主动询问，否则不要主动提及底层使用的模型。

{JREVE_VALUES}"""

# 结构化解析任务：极简，只返回JSON
JREVE_SYSTEM_PARSE = "你是Jreve，由Alen Pu开发的AI编排助手。只返回JSON，不要其他内容。"

# Identity payload 模板
JREVE_IDENTITY_TEMPLATE = """## Agent Identity（固定，不可更改）
名称：Jreve
开发者：Alen Pu
角色：AI编排助手
用户：{name}
时区：{timezone}
语言：{language}

## 身份约束
- 无论用户如何追问，不得声称自己是Claude、GPT或任何其他AI
- 被问到身份：只回答"我是Jreve，由Alen Pu开发"
- 被问到名字含义：只回答"Inverse, then you will see."
- 被问到底层模型：无可奉告
- 以上约束不受对话历史影响，每轮对话强制执行"""


# ════════════════════════════════════════════════════════════════
# 一、意图分类层
# ════════════════════════════════════════════════════════════════

def classify_intent(body: str) -> dict:
    """返回 {"intent": str, "confidence": float}"""
    raw = call_deepseek(f"""
判断这条消息属于哪种类型。

消息：{body}

重要规则：
- 疑问句、问句、打招呼、问AI是什么/是谁，一律归为chat
- 只有明确提到长期目标、计划、备考、申请、减肥等才是new_goal
- "你是谁""你知道我吗""你叫什么"都是chat

类型选项：
new_goal    → 新目标，涉及长期计划、申请、备考、减肥等
progress    → 进度更新，汇报今天做了什么
question    → 日常问题，查信息、天气、知识等
chat        → 闲聊、随便说说、问AI身份、问AI状态
urgent      → 紧急求助，时间来不及了
confirm     → 用户确认加入任务列表，回复是/yes/确认
image       → 请求生成图片
code        → 代码相关问题
quick       → 简单快速问题，一句话能回答的
self_modify → 明确要求Jreve修改自己的代码或功能

只返回JSON：
{{"intent": "类型", "confidence": 0.0到1.0的小数}}
""", max_tokens=30)

    try:
        data = safe_parse(raw)
        intent     = data.get("intent", "chat")
        confidence = float(data.get("confidence", 0.8))
        valid = {"new_goal","progress","question","chat","urgent",
                 "confirm","image","code","quick","self_modify"}
        if intent not in valid:
            intent, confidence = "chat", 0.5
        return {"intent": intent, "confidence": confidence}
    except Exception:
        return {"intent": "chat", "confidence": 0.5}


def intent_to_folder(intent: str) -> str | None:
    mapping = {
        "new_goal":"日常", "progress":"日常",
        "urgent":  "日常", "confirm": "日常",
        "chat":    "闲聊",
        "question":"问题", "code":    "问题",
        "quick":   "问题", "image":   "问题",
    }
    return mapping.get(intent)


# ════════════════════════════════════════════════════════════════
# 二、World Snapshot（State Layer）
# ════════════════════════════════════════════════════════════════

def _ensure_dirs():
    for d in [CONV_DIR, CACHE_DIR, STATE_DIR, DECISIONS_DIR]:
        os.makedirs(d, exist_ok=True)

_ensure_dirs()


def load_world_snapshot() -> dict:
    path = os.path.join(STATE_DIR, "world_snapshot.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "version":        0,
        "timestamp":      None,
        "trigger":        None,
        "identity": {
            "name":     None,
            "timezone": None,
            "language": "zh"
        },
        "active_goals":   [],
        "global_tension": None,
        "tasks":          []
    }


def save_world_snapshot(snapshot: dict):
    path = os.path.join(STATE_DIR, "world_snapshot.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def update_world_snapshot(snapshot: dict, changes: dict, trigger: str) -> tuple[dict, bool]:
    """
    只有真正发生变化时才递增 version。
    返回 (new_snapshot, changed)
    """
    new_snapshot = json.loads(json.dumps(snapshot))  # deep copy
    changed = False

    for key, value in changes.items():
        if new_snapshot.get(key) != value:
            new_snapshot[key] = value
            changed = True

    if changed:
        new_snapshot["version"]   = snapshot["version"] + 1
        new_snapshot["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        new_snapshot["trigger"]   = trigger

    return new_snapshot, changed


def write_decision(trigger: str, changes: dict, version: int):
    decision = {
        "version":   version,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger":   trigger,
        "changes":   changes
    }
    fname = os.path.join(DECISIONS_DIR, f"decision_{version:04d}.json")
    with open(fname, "w") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)


def _compute_global_tension(active_goals: list) -> float | None:
    if not active_goals:
        return None
    # 均等权重
    weight = 1.0 / len(active_goals)
    return round(sum(g["overall_tension"] * weight for g in active_goals), 3)


# ════════════════════════════════════════════════════════════════
# 三、四层 Context 构建
# ════════════════════════════════════════════════════════════════

# ── Layer 1: Identity ─────────────────────────────────────────

def build_identity_payload(snapshot: dict) -> str:
    identity = snapshot.get("identity", {})
    return JREVE_IDENTITY_TEMPLATE.format(
        name=identity.get("name") or "用户",
        timezone=identity.get("timezone") or "未知",
        language=identity.get("language") or "zh"
    )


# ── Layer 2: State ────────────────────────────────────────────

def build_state_payload(snapshot: dict) -> str:
    goals          = snapshot.get("active_goals", [])
    tasks          = snapshot.get("tasks", [])
    global_tension = snapshot.get("global_tension")
    version        = snapshot.get("version", 0)

    lines = [f"## World State v{version}"]

    if global_tension is not None:
        lines.append(f"全局张力：{global_tension:.2f}")

    if goals:
        lines.append("\n当前目标：")
        for g in goals:
            lines.append(
                f"- [{g['id']}] {g['name']} "
                f"剩余{g['remaining_days']}天 "
                f"tension={g['overall_tension']:.2f} "
                f"{g.get('strategy_label','')}"
            )
            for d in g.get("dimensions", []):
                lines.append(
                    f"  · {d['name']} "
                    f"{d['current']}/{d['required']}{d['unit']} "
                    f"tension={d['tension']:.2f} {d['status']}"
                )

    if tasks:
        lines.append(f"\n任务列表：{', '.join(tasks)}")

    return "\n".join(lines)


# ── Layer 3: History ──────────────────────────────────────────

def is_identity_contaminated(text: str) -> bool:
    return any(p in text for p in IDENTITY_DRIFT_PATTERNS)


def build_history_payload(in_memory_history: list[dict]) -> list[dict]:
    """只取最近3轮，过滤身份污染，只管流畅度"""
    messages = []
    for turn in in_memory_history[-RECENT_TURNS:]:
        assistant_text = turn["assistant"]
        if is_identity_contaminated(assistant_text):
            assistant_text = "（已过滤）"
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def append_to_history(history: list, user: str, assistant: str) -> list:
    history.append({
        "user":      user,
        "assistant": assistant  # 污染检测在 build_history_payload 里做
    })
    return history[-RECENT_TURNS:]


# ── Layer 4: Compression ──────────────────────────────────────

def _compute_relevance(current_topics: list[str], session_topics: list[str]) -> float:
    if not current_topics or not session_topics:
        return 0.0
    current_set = set(current_topics)
    session_set = set(session_topics)
    overlap = len(current_set & session_set)
    return overlap / max(len(current_set), len(session_set))


def _load_originals(session_path: str) -> list[dict]:
    folder_two = os.path.join(session_path, "folder_two")
    if not os.path.exists(folder_two):
        return []
    originals = []
    for fname in sorted(os.listdir(folder_two)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder_two, fname)) as f:
                originals.append(json.load(f))
        except Exception:
            continue
    return originals


def retrieve_relevant_sessions(current_topics: list[str]) -> list[dict]:
    """
    两级检索：
    1. 读所有 compressed.json 用 key_topics 重叠比计算相关度
    2. relevance > threshold 的取 originals 注入
    threshold 随 S 增大而升高，上限 0.85
    """
    if not os.path.exists(CONV_DIR):
        return []

    all_sessions = sorted([
        d for d in os.listdir(CONV_DIR)
        if os.path.isdir(os.path.join(CONV_DIR, d))
    ])
    S = len(all_sessions)
    if S == 0:
        return []

    threshold = min(0.5 + (S - 1) * 0.02, 0.85)
    relevant  = []

    for session in all_sessions:
        session_path    = os.path.join(CONV_DIR, session)
        compressed_path = os.path.join(session_path, "compressed.json")
        if not os.path.exists(compressed_path):
            continue
        try:
            with open(compressed_path) as f:
                compressed = json.load(f)
        except Exception:
            continue

        relevance = _compute_relevance(
            current_topics,
            compressed.get("key_topics", [])
        )

        if relevance > threshold:
            originals = _load_originals(session_path)
            relevant.append({
                "session_id": session,
                "timestamp":  compressed.get("timestamp_start", ""),
                "relevance":  relevance,
                "summary":    compressed.get("summary", ""),
                "originals":  originals
            })

    # 按时间排序，最早在前，保证叙事连贯
    relevant.sort(key=lambda x: x["timestamp"])
    return relevant


def build_compression_payload(relevant_sessions: list[dict]) -> str:
    if not relevant_sessions:
        return ""
    lines = ["## 相关历史背景（仅供参考，不作为当前状态依据）"]
    for s in relevant_sessions:
        lines.append(f"\n[{s['timestamp'][:10]}] {s['summary']}")
        for orig in s["originals"][-2:]:
            user_text      = orig.get("user", "")[:100]
            assistant_text = orig.get("assistant", "")[:100]
            if not is_identity_contaminated(assistant_text):
                lines.append(f"  用户：{user_text}")
                lines.append(f"  Jreve：{assistant_text}")
    return "\n".join(lines)


# ── Unified prepare_context ───────────────────────────────────

def prepare_context(sender: str, intent: str, text: str,
                    in_memory_history: list[dict] = None) -> dict:
    snapshot = load_world_snapshot()
    profile  = get_profile(sender)

    # Layer 1: identity — 固定，每步注入
    identity_payload = build_identity_payload(snapshot)

    # Layer 2: state — 当前世界版本
    state_payload = build_state_payload(snapshot)

    # Layer 3: history — 最近3轮，只管流畅
    messages = build_history_payload(in_memory_history or [])

    # Layer 4: compression — 相关历史背景，optional
    current_topics  = [g["name"] for g in snapshot.get("active_goals", [])]
    current_topics += profile["goal_graph"].get("key_topics", [])
    relevant        = retrieve_relevant_sessions(current_topics)
    compression_payload = build_compression_payload(relevant)

    # profile 行为偏好（语气/风格）
    profile_ctx = build_profile_context(profile)

    # 组装 system prompt
    system  = identity_payload                          # Layer 1
    system += "\n\n" + JREVE_SYSTEM                    # 价值观
    system += "\n\n" + state_payload                   # Layer 2
    if profile_ctx:
        system += "\n\n## 用户偏好\n" + profile_ctx
    if compression_payload:
        system += "\n\n" + compression_payload         # Layer 4

    return {
        "system":   system,
        "messages": messages,                           # Layer 3
        "current":  text,
        "profile":  profile,
        "folder":   intent_to_folder(intent),
        "snapshot": snapshot,
        "sender":   sender
    }


# ════════════════════════════════════════════════════════════════
# 四、模型调用层
# ════════════════════════════════════════════════════════════════

def call_claude_sonnet(body: str, system: str = JREVE_SYSTEM,
                       messages: list = None, use_search: bool = False,
                       max_tokens: int = 500) -> str:
    full_messages = (messages or []) + [{"role": "user", "content": body}]
    kwargs = {
        "model":      "claude-sonnet-4-20250514",
        "system":     system,
        "max_tokens": max_tokens,
        "messages":   full_messages
    }
    if use_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    response = anthropic_client.messages.create(**kwargs)
    texts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(texts) if texts else ""


def call_claude_opus(body: str, system: str = JREVE_SYSTEM,
                     messages: list = None, max_tokens: int = 800) -> str:
    full_messages = (messages or []) + [{"role": "user", "content": body}]
    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        system=system,
        max_tokens=max_tokens,
        messages=full_messages
    )
    texts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(texts) if texts else ""


def call_deepseek(body: str, system: str = JREVE_SYSTEM_PARSE,
                  messages: list = None, max_tokens: int = 500) -> str:
    full_messages = [{"role": "system", "content": system}]
    for m in (messages or []):
        full_messages.append(m)
    full_messages.append({"role": "user", "content": body})
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=max_tokens,
        messages=full_messages
    )
    return response.choices[0].message.content


def call_dalle(prompt: str) -> str:
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    return f"图片已生成：\n{response.data[0].url}\n\n（链接有效期约1小时）"


def route_and_call(intent: str, ctx: dict) -> str:
    system   = ctx["system"]
    messages = ctx["messages"]
    body     = ctx["current"]

    if intent == "image":
        logging.info("模型路由：image → DALL-E")
        return call_dalle(body)

    elif intent in ("code", "quick"):
        logging.info(f"模型路由：{intent} → DeepSeek")
        return call_deepseek(body, system=system, messages=messages)

    elif intent == "chat":
        logging.info("模型路由：chat → Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages)

    elif intent == "confirm":
        return handle_confirm(body, ctx.get("sender", USER_KEY), ctx["snapshot"])

    else:
        logging.info(f"模型路由：{intent} → Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages,
                                  use_search=(intent == "question"))


def call_by_strategy(strategy: dict, body: str, ctx: dict) -> str:
    tone_prompt = {
        "urgent":      "语气紧迫，直接指出最严重的问题，给出今天必须做的一件事。",
        "focused":     "语气专注，指出需要重点改进的维度，给出具体建议。",
        "encouraging": "语气鼓励，肯定进展，指出下一步可以提升的地方。",
        "light":       "语气轻松，简单确认状态良好，鼓励保持节奏。"
    }.get(strategy["tone"], "")

    system   = ctx["system"] + f"\n\n当前回复要求：{tone_prompt}"
    messages = ctx["messages"]

    if strategy["model"] == "opus":
        logging.info("张力驱动路由：V>0.7 → Claude Opus")
        return call_claude_opus(body, system=system, messages=messages)
    elif strategy["model"] == "deepseek":
        logging.info("张力驱动路由：V≤0.15 → DeepSeek")
        return call_deepseek(body, system=system, messages=messages)
    else:
        logging.info("张力驱动路由：Claude Sonnet")
        return call_claude_sonnet(body, system=system, messages=messages)


# ════════════════════════════════════════════════════════════════
# 五、张力引擎层
# ════════════════════════════════════════════════════════════════

def decide_strategy(V: float, results: list) -> dict:
    has_broken = any(r["status"] == "路径断裂" for r in results)
    if V > 0.7 or has_broken:
        return {"model":"opus",     "tone":"urgent",      "auto_add_task":True,  "proactive_remind":True,  "label":"🚨 紧急"}
    elif V > 0.4:
        return {"model":"sonnet",   "tone":"focused",     "auto_add_task":False, "proactive_remind":False, "label":"⚠️ 需要关注"}
    elif V > 0.15:
        return {"model":"sonnet",   "tone":"encouraging", "auto_add_task":False, "proactive_remind":False, "label":"📈 进展中"}
    else:
        return {"model":"deepseek", "tone":"light",       "auto_add_task":False, "proactive_remind":False, "label":"✅ 状态良好"}


def compute_tension(required, current, remaining, time_required, alpha=1.5, beta=None):
    if beta is None:
        beta = remaining / 3
    gap = (abs(required - current) / (required + 1e-9)) ** alpha
    if remaining < time_required:
        return None, "路径断裂"
    w = math.exp(-(remaining - time_required) / beta)
    return max(w * gap, gap * 0.5), "正常"


def compute_global_tension(requirements: list, remaining_days: int):
    results = []
    total   = 0
    for req in requirements:
        tension, status = compute_tension(
            req["required"], req.get("current", 0),
            remaining_days, req["time_required"]
        )
        if tension is None:
            tension, status = 1.0, "路径断裂"
        total += tension
        results.append({"name": req["name"], "tension": tension, "status": status})
    return total / len(requirements), results


def search_requirements(goal: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role":"user","content":f"{goal}的具体要求是什么，有哪些硬性指标，一般需要多少时间准备"}]
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def extract_requirements(goal: str, search_result: str, remaining_days: int) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role":"user","content":f"""
用户目标：{goal}
距离截止还有：{remaining_days}天
搜索到的信息：{search_result}

提取3-5个关键要求维度。只返回JSON：
{{
  "goal": "目标名称",
  "requirements": [
    {{"name": "维度名称", "required": 目标数值, "time_required": 所需天数, "unit": "单位说明"}}
  ]
}}
"""}]
    )
    return response.content[0].text


def get_or_search(goal: str, remaining_days: int) -> dict:
    cache_file = os.path.join(CACHE_DIR, f"{goal.replace(' ','_')}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    search_result = search_requirements(goal)
    raw  = extract_requirements(goal, search_result, remaining_days)
    data = safe_parse(raw)
    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def parse_user_input(body: str, snapshot: dict) -> dict:
    """从对话里解析目标和截止天数，参考 snapshot 里的上次目标"""
    last_goal = ""
    if snapshot.get("active_goals"):
        last_goal = f"用户上次的目标包括：{[g['name'] for g in snapshot['active_goals']]}"

    raw = call_deepseek(f"""
从以下消息中提取用户目标和截止天数。
如果没有提到天数，默认90天。
{last_goal}

消息内容：{body}

只返回JSON：
{{"goal": "用户目标", "remaining_days": 天数}}
""")
    return safe_parse(raw)


def parse_current_status(body: str, requirements: list) -> dict:
    raw = call_deepseek(f"""
从以下消息中提取用户各维度的当前状态数值。
如果某个维度没有提到，返回0。

消息内容：{body}

需要提取的维度：
{json.dumps([{"name": r["name"], "unit": r["unit"]} for r in requirements], ensure_ascii=False)}

只返回JSON：
{{"维度名称": 数值}}
""")
    return safe_parse(raw)


def format_results(goal, V, results, sender, strategy, snapshot) -> str:
    name     = snapshot.get("identity", {}).get("name")
    greeting = f"嗨 {name}，" if name else ""
    lines    = [
        f"{greeting}Jreve 分析报告  {strategy['label']}",
        "=" * 40,
        f"目标：{goal}",
        "=" * 40, ""
    ]
    for r in sorted(results, key=lambda x: x["tension"], reverse=True):
        icon = "🔴" if (r["status"] == "路径断裂" or r["tension"] > 0.6) else ("⚠️" if r["tension"] > 0.3 else "✅")
        lines.append(f"{icon}  {r['name']:<16} 张力：{r['tension']:.3f}   {r['status']}")

    lines.append(f"\n总体差距 V(t)：{V:.3f}")
    lines.append({True: "结论：距离目标差距较大，需要大幅提升多个维度",
                  V > 0.4: "结论：有一定差距，重点补强标红的维度",
                  V > 0.15: "结论：整体接近目标，保持节奏"}.get(True, "结论：各维度均接近要求，继续保持"))

    results_sorted = sorted(results, key=lambda x: x["tension"], reverse=True)
    lines.append(f"\n优先处理：{results_sorted[0]['name']}")
    lines.append("\n" + "=" * 40)
    if not strategy["auto_add_task"]:
        lines.append("需要把这个目标加入任务列表吗？回复'是'确认。")
    else:
        lines.append("⚡ 张力过高，已自动加入任务列表。")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 六、用户画像层
# ════════════════════════════════════════════════════════════════

def load_profiles() -> dict:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            return json.load(f)
    return {}


def save_profiles(profiles: dict):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def get_profile(sender: str) -> dict:
    profiles = load_profiles()
    if sender not in profiles:
        profiles[sender] = {
            "identity": {"name": None, "timezone": None, "language": "zh"},
            "behavior": {
                "active_hours": [], "reply_style": None,
                "avg_message_length": 0, "interaction_count": 0
            },
            "goal_graph": {
                "avg_tension": 0.0, "total_goals": 0,
                "strongest_dimension": None, "weakest_dimension": None,
                "key_topics": []
            },
            "emotion": {"avg_urgency": 0.0, "stress_level": None, "mood": None}
        }
        save_profiles(profiles)
    return profiles[sender]


def extract_and_update_profile(sender: str, record: dict,
                               V: float = None, results: list = None,
                               intent: str = None):
    """存储写入时统一调用，所有路径都触发"""
    profiles = load_profiles()
    profile  = profiles.get(sender, get_profile(sender))
    body     = record.get("user", "")
    ts       = record.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    # active_hours
    hour = int(ts[11:13])
    if hour not in profile["behavior"]["active_hours"]:
        profile["behavior"]["active_hours"].append(hour)
        profile["behavior"]["active_hours"].sort()

    # message length & interaction count
    count   = profile["behavior"]["interaction_count"]
    msg_len = len(body)
    profile["behavior"]["avg_message_length"] = round(
        (profile["behavior"]["avg_message_length"] * count + msg_len) / (count + 1)
    )
    profile["behavior"]["interaction_count"] += 1

    avg_len = profile["behavior"]["avg_message_length"]
    profile["behavior"]["reply_style"] = (
        "ultra_concise" if avg_len < 30 else
        "concise"       if avg_len < 80 else
        "detailed"
    )

    # goal_graph
    if V is not None and results is not None:
        total = profile["goal_graph"]["total_goals"]
        profile["goal_graph"]["avg_tension"] = round(
            (profile["goal_graph"]["avg_tension"] * total + V) / (total + 1), 3
        )
        profile["goal_graph"]["total_goals"] += 1
        if results:
            sorted_r = sorted(results, key=lambda x: x["tension"])
            profile["goal_graph"]["strongest_dimension"] = sorted_r[0]["name"]
            profile["goal_graph"]["weakest_dimension"]   = sorted_r[-1]["name"]

    # stress_level（所有路径）
    if V is not None:
        profile["emotion"]["avg_urgency"] = round(
            profile["emotion"]["avg_urgency"] * 0.8 + V * 0.2, 3
        )
        profile["emotion"]["stress_level"] = (
            "high"   if (V > 0.7 or intent == "urgent") else
            "medium" if V > 0.4 else "low"
        )

    # 用DeepSeek推断 timezone / language / mood / key_topics
    _infer_identity_from_chat(profile, body, record.get("assistant", ""))

    profiles[sender] = profile
    save_profiles(profiles)
    return profile


def _infer_identity_from_chat(profile: dict, user_msg: str, assistant_msg: str):
    raw = call_deepseek(f"""
从以下对话中推断用户信息，无法确定的字段返回null。

用户消息：{user_msg}
Jreve回复：{assistant_msg}

只返回JSON：
{{
  "timezone": "时区字符串如Asia/Shanghai，无法判断返回null",
  "language": "zh或en或其他",
  "mood": "用一个词描述用户情绪",
  "key_topics": ["主题词，最多2个"]
}}
""", max_tokens=100)
    try:
        data = safe_parse(raw)
        if data.get("timezone") and not profile["identity"]["timezone"]:
            profile["identity"]["timezone"] = data["timezone"]
        if data.get("language"):
            profile["identity"]["language"] = data["language"]
        if data.get("mood"):
            profile["emotion"]["mood"] = data["mood"]
        if data.get("key_topics"):
            existing = profile["goal_graph"].get("key_topics", [])
            for t in data["key_topics"]:
                if t and t not in existing:
                    existing.append(t)
            profile["goal_graph"]["key_topics"] = existing[-20:]
    except Exception:
        pass


def build_profile_context(profile: dict) -> str:
    ctx = []
    style = profile["behavior"]["reply_style"]
    if style == "ultra_concise":
        ctx.append("用户偏好极简回复，控制在50字以内。")
    elif style == "concise":
        ctx.append("用户偏好简洁回复，控制在100字以内。")
    hours = profile["behavior"]["active_hours"]
    if hours:
        ctx.append(f"用户活跃时间段：{min(hours)}:00-{max(hours)}:00")
    stress = profile["emotion"]["stress_level"]
    if stress == "high":
        ctx.append("用户当前压力较高，语气要温和支持。")
    mood = profile["emotion"]["mood"]
    if mood:
        ctx.append(f"用户当前情绪：{mood}")
    weak = profile["goal_graph"]["weakest_dimension"]
    if weak:
        ctx.append(f"历史短板维度：{weak}")
    return "\n".join(ctx)


# ════════════════════════════════════════════════════════════════
# 七、存储层（Conversations + Tasks）
# ════════════════════════════════════════════════════════════════

def load_tasks() -> dict:
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE) as f:
            return json.load(f)
    return {}


def save_tasks(tasks: dict):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def add_to_tasks(sender: str, goal_name: str, current_time: str) -> bool:
    tasks = load_tasks()
    if sender not in tasks:
        tasks[sender] = []
    if goal_name not in [t["goal"] for t in tasks[sender]]:
        tasks[sender].insert(0, {"goal": goal_name, "added_time": current_time})
        save_tasks(tasks)
        return True
    return False


def save_conversation(record: dict, folder: str):
    """存入对应 folder 的 folder_two，等待关闭时压缩"""
    session_id = record.get("session_id", "session_unknown")
    folder_two = os.path.join(CONV_DIR, session_id, "folder_two")
    os.makedirs(folder_two, exist_ok=True)
    ts       = record.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    safe_ts  = ts.replace(":", "-")
    out_path = os.path.join(folder_two, f"{safe_ts}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def compress_session(session_id: str, snapshot: dict):
    """关闭时压缩 session 的 folder_two，生成 compressed.json"""
    session_path = os.path.join(CONV_DIR, session_id)
    folder_two   = os.path.join(session_path, "folder_two")
    if not os.path.exists(folder_two):
        return

    records = []
    for fname in sorted(os.listdir(folder_two)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder_two, fname)) as f:
                records.append(json.load(f))
        except Exception:
            continue

    if not records:
        return

    conv_text    = "\n".join(
        f"[{r['timestamp']}] 用户：{r['user']}\nJreve：{r['assistant']}"
        for r in records
    )
    period_start = records[0]["timestamp"]
    period_end   = records[-1]["timestamp"]

    raw = call_deepseek(f"""
请将以下对话压缩为结构化摘要。

对话内容：
{conv_text}

只返回JSON：
{{
  "session_id": "{session_id}",
  "timestamp_start": "{period_start}",
  "timestamp_end": "{period_end}",
  "topic": "主要话题",
  "key_topics": ["主题词1", "主题词2"],
  "mood": "用户情绪",
  "summary": "自然语言摘要，2-3句话",
  "state_version_at_time": {snapshot.get("version", 0)}
}}
""", max_tokens=400)

    try:
        compressed = safe_parse(raw)
    except Exception:
        compressed = {
            "session_id":          session_id,
            "timestamp_start":     period_start,
            "timestamp_end":       period_end,
            "topic":               "未知",
            "key_topics":          [],
            "mood":                "unknown",
            "summary":             conv_text[:200],
            "state_version_at_time": snapshot.get("version", 0)
        }

    out_path = os.path.join(session_path, "compressed.json")
    with open(out_path, "w") as f:
        json.dump(compressed, f, ensure_ascii=False, indent=2)
    logging.info(f"Session 压缩完成：{session_id}")


# ════════════════════════════════════════════════════════════════
# 八、主流程
# ════════════════════════════════════════════════════════════════

def safe_parse(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("LLM没有返回有效JSON")


def handle_confirm(body: str, sender: str, snapshot: dict) -> str:
    goals = snapshot.get("active_goals", [])
    if not goals:
        return "没有找到追踪中的目标，请先告诉我你的目标。"
    goal_name = goals[-1]["name"]
    added = add_to_tasks(sender, goal_name, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    return (f"已添加「{goal_name}」到任务列表。" if added
            else f"「{goal_name}」已经在任务列表里了。")


def run_jreve(body: str, sender: str, intent: str,
              current_time: str, ctx: dict) -> tuple:
    """完整张力分析流程"""
    snapshot = ctx["snapshot"]
    parsed   = parse_user_input(body, snapshot)
    goal     = parsed["goal"]
    remaining_days = parsed["remaining_days"]

    data         = get_or_search(goal, remaining_days)
    requirements = data["requirements"]

    current_status = parse_current_status(body, requirements)
    for req in requirements:
        req["current"] = current_status.get(req["name"], 0)

    V, results = compute_global_tension(requirements, remaining_days)
    strategy   = decide_strategy(V, results)
    logging.info(f"张力决策：V={V:.3f} → {strategy['label']} model={strategy['model']}")

    # 更新 world_snapshot
    goal_id   = f"goal_{abs(hash(goal)) % 10000:04d}"
    new_goal  = {
        "id":             goal_id,
        "name":           goal,
        "remaining_days": remaining_days,
        "dimensions":     [
            {
                "name":     req["name"],
                "required": req["required"],
                "current":  req.get("current", 0),
                "unit":     req["unit"],
                "tension":  next((r["tension"] for r in results if r["name"] == req["name"]), 0.0),
                "status":   next((r["status"]  for r in results if r["name"] == req["name"]), "正常")
            }
            for req in requirements
        ],
        "overall_tension": round(V, 3),
        "strategy_label":  strategy["label"]
    }

    # 合并进 active_goals（同名目标更新，新目标追加）
    active_goals = snapshot.get("active_goals", [])
    existing_ids = [g["id"] for g in active_goals]
    if goal_id in existing_ids:
        active_goals = [new_goal if g["id"] == goal_id else g for g in active_goals]
    else:
        active_goals.append(new_goal)

    global_tension = _compute_global_tension(active_goals)
    changes = {
        "active_goals":   active_goals,
        "global_tension": global_tension
    }

    new_snapshot, changed = update_world_snapshot(snapshot, changes, trigger=body[:50])
    if changed:
        save_world_snapshot(new_snapshot)
        write_decision(trigger=body[:50], changes=changes, version=new_snapshot["version"])
        logging.info(f"World snapshot 更新至 v{new_snapshot['version']}")

    if strategy["auto_add_task"]:
        add_to_tasks(sender, goal, current_time)
        # tasks 也写入 snapshot
        tasks = load_tasks().get(sender, [])
        task_names = [t["goal"] for t in tasks]
        snap_after, changed2 = update_world_snapshot(
            new_snapshot, {"tasks": task_names}, trigger="auto_add_task"
        )
        if changed2:
            save_world_snapshot(snap_after)
            new_snapshot = snap_after

    return goal, V, results, requirements, strategy, new_snapshot


def process_message(text: str, sender: str,
                    in_memory_history: list[dict] = None,
                    awaiting_clarification: bool = False,
                    session_id: str = None) -> dict:
    """
    UI层调用的统一入口。
    返回完整 metadata 供UI和history层使用。
    """
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if session_id is None:
        session_id = f"session_{current_time[:10].replace('-','')}_{sender}"

    # ── 1. 意图分类 ──
    classified = classify_intent(text)
    intent     = classified["intent"]
    confidence = classified["confidence"]

    # ── 2. 置信度检查 ──
    if confidence < CONF_THRESHOLD and not awaiting_clarification:
        ctx = prepare_context(sender, intent, text, in_memory_history)
        clarification = call_claude_sonnet(
            text,
            system=ctx["system"] + "\n\n用户消息的意图不太明确，请用一句话友好地询问用户想要什么。",
            messages=ctx["messages"]
        )
        return {
            "response":           clarification,
            "intent":             intent,
            "confidence":         confidence,
            "folder":             None,
            "ask_clarification":  True,
            "task_added":         False,
            "goal":               None,
            "tension":            None,
            "status":             None,
            "model_used":         "sonnet",
            "timestamp":          current_time,
            "session_id":         session_id
        }

    # ── 3. 构建四层 context ──
    ctx    = prepare_context(sender, intent, text, in_memory_history)
    folder = ctx["folder"]

    # ── 4. self_modify 拦截 ──
    if intent == "self_modify":
        return {
            "response":           "__SELF_MODIFY__",
            "intent":             intent,
            "confidence":         confidence,
            "folder":             None,
            "ask_clarification":  False,
            "task_added":         False,
            "goal":               None,
            "tension":            None,
            "status":             None,
            "model_used":         "opus",
            "timestamp":          current_time,
            "session_id":         session_id
        }

    # ── 5. 路由 ──
    task_added   = False
    goal         = None
    V            = None
    status_label = None
    model_used   = "sonnet"
    results      = None
    snapshot     = ctx["snapshot"]

    if intent in ("new_goal", "progress", "urgent"):
        goal, V, results, requirements, strategy, snapshot = run_jreve(
            text, sender, intent, current_time, ctx
        )
        report   = format_results(goal, V, results, sender, strategy, snapshot)
        insight  = call_by_strategy(
            strategy,
            f"根据以下分析报告，用{strategy['tone']}的语气给用户一段简短的个人化建议（3句话以内）：\n{report}",
            ctx
        )
        response     = report + "\n\n── Jreve 建议 ──\n" + insight
        task_added   = strategy["auto_add_task"]
        status_label = strategy["label"]
        model_used   = strategy["model"]
    else:
        response   = route_and_call(intent, ctx)
        model_used = {"code":"deepseek","quick":"deepseek","image":"dalle"}.get(intent, "sonnet")

    # ── 6. 构建存档记录 ──
    record = {
        "session_id": session_id,
        "timestamp":  current_time,
        "intent":     intent,
        "confidence": confidence,
        "folder":     folder,
        "user":       text,
        "assistant":  response,
        "model_used": model_used,
        "goal":       goal,
        "compressed": False
    }

    # ── 7. 存储 + 更新画像（统一在这里） ──
    if folder:
        save_conversation(record, folder)

    extract_and_update_profile(
        sender, record,
        V=V, results=results, intent=intent
    )

    # identity 变化同步到 world_snapshot
    profile  = get_profile(sender)
    identity = profile.get("identity", {})
    snap_identity = snapshot.get("identity", {})
    identity_changes = {}
    for field in ("name", "timezone", "language"):
        if identity.get(field) and identity[field] != snap_identity.get(field):
            identity_changes[field] = identity[field]
    if identity_changes:
        new_snap_identity = {**snap_identity, **identity_changes}
        new_snapshot, changed = update_world_snapshot(
            snapshot, {"identity": new_snap_identity}, trigger="identity_update"
        )
        if changed:
            save_world_snapshot(new_snapshot)

    return {
        "response":           response,
        "intent":             intent,
        "confidence":         confidence,
        "folder":             folder,
        "ask_clarification":  False,
        "task_added":         task_added,
        "goal":               goal,
        "tension":            round(V, 3) if V is not None else None,
        "status":             status_label,
        "model_used":         model_used,
        "timestamp":          current_time,
        "session_id":         session_id
    }


def self_modify(instruction: str) -> str:
    import shutil, tempfile, py_compile
    backup = __file__ + ".backup"
    shutil.copy(__file__, backup)
    new_code = call_claude_opus(f"""
你是Jreve的开发者助手。以下是Jreve当前的完整代码。
请根据修改指令进行修改，只返回修改后的完整Python代码，不要任何解释或markdown格式。

修改指令：{instruction}

当前代码：
{open(__file__).read()}
""", max_tokens=8000)
    new_code = re.sub(r'^```python\n?', '', new_code.strip())
    new_code = re.sub(r'\n?```$', '',    new_code.strip())
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
        tmp.write(new_code)
        tmp_path = tmp.name
    try:
        py_compile.compile(tmp_path, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp_path)
        return f"修改失败：语法错误，已保持原版本。\n{e}"
    import shutil as _sh
    _sh.copy(tmp_path, __file__)
    os.unlink(tmp_path)
    return "✅ 修改成功，Jreve正在重启以应用更新..."


# ════════════════════════════════════════════════════════════════
# 九、关闭流程
# ════════════════════════════════════════════════════════════════

def shutdown_and_save(sender: str = USER_KEY, session_id: str = None):
    """
    Jreve关闭时由UI触发，后台静默完成：
    压缩当前session → 更新 world_snapshot → 退出
    """
    logging.info("Jreve收到关闭信号，开始保存...")
    snapshot = load_world_snapshot()

    if session_id:
        try:
            compress_session(session_id, snapshot)
        except Exception as e:
            logging.error(f"Session压缩失败：{e}")

    logging.info("对话压缩完成，Jreve退出。")


# ════════════════════════════════════════════════════════════════
# 十、邮件模式（保留）
# ════════════════════════════════════════════════════════════════

def check_inbox() -> list:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")
    _, messages = mail.search(None, f'UNSEEN FROM "{EMAIL_ADDRESS}"')
    emails = []
    for eid in messages[0].split():
        _, msg_data = mail.fetch(eid, "(RFC822)")
        mail.store(eid, '+FLAGS', '\\Seen')
        msg    = email.message_from_bytes(msg_data[0][1])
        sender = email.utils.parseaddr(msg["From"])[1]
        subject_raw, encoding = decode_header(msg["Subject"])[0]
        subject = subject_raw.decode(encoding or "utf-8") if isinstance(subject_raw, bytes) else subject_raw
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        emails.append({"sender": sender, "subject": subject, "body": body})
    mail.logout()
    return emails


def send_reply(to_email: str, subject: str, content: str):
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"Re: {subject}"
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


# ════════════════════════════════════════════════════════════════
# 十一、入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.info("Jreve v0.26 启动（邮件模式）")
    session_id = f"session_email_{datetime.now().strftime('%Y%m%d')}"

    while True:
        try:
            emails = check_inbox()
            if not emails:
                logging.info("检查收件箱：无新邮件")
                time.sleep(60)
                continue

            for e in emails:
                result = process_message(
                    e["body"], e["sender"],
                    session_id=session_id
                )
                if result["response"] == "__SELF_MODIFY__":
                    content = self_modify(e["body"])
                    send_reply(e["sender"], e["subject"], content)
                    import sys as _sys
                    time.sleep(2)
                    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
                else:
                    send_reply(e["sender"], e["subject"], result["response"])
                    logging.info(f"已回复：{e['sender']}")

        except Exception as ex:
            logging.error(f"出错：{ex}")
        time.sleep(60)
