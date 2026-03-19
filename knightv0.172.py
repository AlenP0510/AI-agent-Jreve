import math
import json
import re
import os
import time
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EMAIL_ADDRESS = os.environ.get("KNIGHT_EMAIL")
EMAIL_PASSWORD = os.environ.get("KNIGHT_EMAIL_PASSWORD")


def safe_parse(raw):
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("LLM没有返回有效JSON")


def compute_tension(required, current, remaining, time_required, alpha=1.5, beta=None):
    if beta is None:
        beta = remaining / 3
    gap = (abs(required - current) / (required + 1e-9)) ** alpha
    if remaining < time_required:
        return None, "路径断裂"
    w = math.exp(-(remaining - time_required) / beta)
    tension = max(w * gap, gap * 0.5)
    return tension, "正常"


def search_requirements(goal):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"{goal}的具体要求是什么，有哪些硬性指标，一般需要多少时间准备"
        }]
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def extract_requirements(goal, search_result, remaining_days):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""
用户目标：{goal}
距离截止还有：{remaining_days}天
搜索到的信息：{search_result}

提取3-5个关键要求维度。
time_required：准备这个维度需要多少天。
name必须是用户能直接理解的日常语言。
unit必须是用户填写时能直接理解的单位。

只返回JSON，不要其他内容：
{{
  "goal": "目标名称",
  "requirements": [
    {{"name": "维度名称", "required": 目标数值, "time_required": 所需天数, "unit": "单位说明"}}
  ]
}}
"""
        }]
    )
    return response.content[0].text


def get_or_search(goal, remaining_days):
    cache_file = f"cache/{goal.replace(' ', '_')}.json"

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    search_result = search_requirements(goal)
    raw = extract_requirements(goal, search_result, remaining_days)
    data = safe_parse(raw)

    os.makedirs("cache", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    return data


def parse_user_email(body):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""
从以下邮件正文中提取用户目标和截止天数。
如果没有提到天数，默认90天。

邮件内容：{body}

只返回JSON：
{{"goal": "用户目标", "remaining_days": 天数, "current_status": {{}}}}
"""
        }]
    )
    return safe_parse(response.content[0].text)


def parse_current_status(body, requirements):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""
从以下邮件正文中提取用户各维度的当前状态数值。
如果某个维度没有提到，返回0。

邮件内容：{body}

需要提取的维度：
{json.dumps([{"name": r["name"], "unit": r["unit"]} for r in requirements], ensure_ascii=False)}

只返回JSON，格式为维度名称对应数值：
{{"维度名称": 数值}}
"""
        }]
    )
    return safe_parse(response.content[0].text)


def compute_global_tension(requirements, remaining_days):
    results = []
    total_tension = 0

    for req in requirements:
        tension, status = compute_tension(
            required=req["required"],
            current=req.get("current", 0),
            remaining=remaining_days,
            time_required=req["time_required"]
        )

        if tension is None:
            tension = 1.0
            status = "路径断裂"

        total_tension += tension
        results.append({
            "name": req["name"],
            "tension": tension,
            "status": status
        })

    V = total_tension / len(requirements)
    return V, results


def format_results(goal, V, results, requirements):
    lines = []
    lines.append("Knight 分析报告")
    lines.append("=" * 40)
    lines.append(f"目标：{goal}")
    lines.append("=" * 40)
    lines.append("")

    results_sorted = sorted(results, key=lambda x: x["tension"], reverse=True)
    for r in results_sorted:
        if r["status"] == "路径断裂":
            icon = "🔴"
        elif r["tension"] > 0.6:
            icon = "🔴"
        elif r["tension"] > 0.3:
            icon = "⚠️"
        else:
            icon = "✅"
        lines.append(f"{icon}  {r['name']:<16} 张力：{r['tension']:.3f}   {r['status']}")

    lines.append(f"\n总体差距 V(t)：{V:.3f}")

    if V > 0.7:
        lines.append("结论：距离目标差距较大，需要大幅提升多个维度")
    elif V > 0.4:
        lines.append("结论：有一定差距，重点补强标红的维度")
    elif V > 0.15:
        lines.append("结论：整体接近目标，保持节奏")
    else:
        lines.append("结论：各维度均接近要求，继续保持")

    lines.append(f"\n优先处理：{results_sorted[0]['name']}")
    lines.append("")
    lines.append("=" * 40)
    lines.append("回复此邮件以更新进度，Knight会重新计算张力。")

    return "\n".join(lines)


def classify_intent(body):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""
判断这条消息属于哪种类型，只返回一个词，不要其他内容。

消息：{body}

类型选项：
new_goal    → 新目标，涉及长期计划、申请、备考、减肥等
progress    → 进度更新，汇报今天做了什么
question    → 日常问题，查信息、天气、知识等
chat        → 闲聊、随便说说
urgent      → 紧急求助，时间来不及了
"""
        }]
    )
    result = response.content[0].text.strip()
    valid = {"new_goal", "progress", "question", "chat", "urgent"}
    return result if result in valid else "new_goal"


def handle_question(body):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system="你是Knight，一个由Alen Pu开发的AI编排助手。你会根据用户需求调度最合适的模型，帮助用户追踪长期目标、计算差距、制定优先级。除非用户主动询问，否则不要主动提及底层使用的模型或Anthropic。",
        max_tokens=500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": body}]
    )
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    return "\n".join(texts) if texts else ""


def handle_chat(body):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": body}]
    )
    return response.content[0].text


def run_knight(body):
    parsed = parse_user_email(body)
    goal = parsed["goal"]
    remaining_days = parsed["remaining_days"]
    data = get_or_search(goal, remaining_days)
    requirements = data["requirements"]

    current_status = parse_current_status(body, requirements)
    for req in requirements:
        req["current"] = current_status.get(req["name"], 0)

    V, results = compute_global_tension(requirements, remaining_days)
    return goal, V, results, requirements


def check_inbox():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    mail.select("inbox")

    _, messages = mail.search(None, f'UNSEEN FROM "{EMAIL_ADDRESS}"')
    email_ids = messages[0].split()

    emails = []
    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        sender = email.utils.parseaddr(msg["From"])[1]
        subject_raw, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject_raw, bytes):
            subject = subject_raw.decode(encoding or "utf-8")
        else:
            subject = subject_raw

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


def send_reply(to_email, subject, content):
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


# 主程序
print("Knight 邮件服务启动...")
print(f"监听邮箱：{EMAIL_ADDRESS}")
print("每60秒检查一次新邮件\n")

while True:
    try:
        emails = check_inbox()
        if emails:
            print(f"收到 {len(emails)} 封新邮件")
        for e in emails:
            print(f"处理来自 {e['sender']} 的邮件...")
            intent = classify_intent(e["body"])
            print(f"意图分类：{intent}")

            if intent in ("new_goal", "progress", "urgent"):
                goal, V, results, requirements = run_knight(e["body"])
                content = format_results(goal, V, results, requirements)
            elif intent == "question":
                content = handle_question(e["body"])
            else:
                content = handle_chat(e["body"])

            send_reply(e["sender"], e["subject"], content)
            print(f"已回复 {e['sender']}")
    except Exception as ex:
        print(f"出错：{ex}")

    time.sleep(60)
