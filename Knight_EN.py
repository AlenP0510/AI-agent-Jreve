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

# Initialize Anthropic client and email credentials from environment variables
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EMAIL_ADDRESS = os.environ.get("KNIGHT_EMAIL")
EMAIL_PASSWORD = os.environ.get("KNIGHT_EMAIL_PASSWORD")


def safe_parse(raw):
    # Extract JSON from LLM response, even if surrounded by extra text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON returned by LLM")


def compute_tension(required, current, remaining, time_required, alpha=1.5, beta=None):
    # Core tension formula: measures gap between current state and target
    # alpha > 1 amplifies large gaps (non-linear scaling)
    # beta controls time decay — higher beta = slower urgency buildup
    if beta is None:
        beta = remaining / 3

    # gap = normalized distance from current to required, raised to alpha
    gap = (abs(required - current) / (required + 1e-9)) ** alpha

    # Path broken: not enough time left to meet this requirement
    if remaining < time_required:
        return None, "path broken"

    # Time weight: urgency increases exponentially as deadline approaches
    w = math.exp(-(remaining - time_required) / beta)

    # Floor at 50% of raw gap — prevents time from masking real shortfalls
    tension = max(w * gap, gap * 0.5)
    return tension, "on track"


def search_requirements(goal):
    # Use Claude's web search tool to find real-world requirements for the goal
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": f"What are the specific requirements to achieve: {goal}? What are the hard metrics and how long does each typically take to prepare?"
        }]
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def extract_requirements(goal, search_result, remaining_days):
    # Parse raw search text into structured requirement dimensions (JSON)
    # Each dimension has: name, required value, time needed, and unit
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""
User goal: {goal}
Days until deadline: {remaining_days}
Search results: {search_result}

Extract 3-5 key requirement dimensions.
time_required: how many days to prepare this dimension.
name must be plain everyday language the user can understand directly.
unit must be a unit the user can fill in directly, e.g. "times/week", "score out of 100".

Return only JSON, no other content:
{{
  "goal": "goal name",
  "requirements": [
    {{"name": "dimension name", "required": target value, "time_required": days needed, "unit": "unit description"}}
  ]
}}
"""
        }]
    )
    return response.content[0].text


def get_or_search(goal, remaining_days):
    # Cache layer: avoid redundant API calls for the same goal
    # First call: searches and stores result in local cache folder
    # Subsequent calls: reads from cache directly (faster + consistent)
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
    # Extract goal and deadline from freeform email text
    # Defaults to 90 days if no deadline is mentioned
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""
Extract the user's goal and deadline from the following email.
If no deadline is mentioned, default to 90 days.

Email content: {body}

Return only JSON:
{{"goal": "user goal", "remaining_days": days, "current_status": {{}}}}
"""
        }]
    )
    return safe_parse(response.content[0].text)


def parse_current_status(body, requirements):
    # Extract current numeric values for each requirement dimension from the email
    # Returns 0 for any dimension not mentioned by the user
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""
Extract the user's current status for each dimension from the email below.
If a dimension is not mentioned, return 0.

Email content: {body}

Dimensions to extract:
{json.dumps([{"name": r["name"], "unit": r["unit"]} for r in requirements], ensure_ascii=False)}

Return only JSON in the format dimension name: value:
{{"dimension name": value}}
"""
        }]
    )
    return safe_parse(response.content[0].text)


def compute_global_tension(requirements, remaining_days):
    # Compute tension for each dimension, then average across all dimensions
    # V(t) = (1/N) * sum of all T_i  — no priority weighting, all dimensions equal
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
            # Path broken: treat as maximum tension
            tension = 1.0
            status = "path broken"

        total_tension += tension
        results.append({
            "name": req["name"],
            "tension": tension,
            "status": status
        })

    # Global tension field V(t): simple average across all dimensions
    V = total_tension / len(requirements)
    return V, results


def format_results(goal, V, results, requirements):
    # Format tension analysis as a readable email report
    # Sorted by tension descending — highest priority item shown first
    lines = []
    lines.append("Knight Analysis Report")
    lines.append("=" * 40)
    lines.append(f"Goal: {goal}")
    lines.append("=" * 40)
    lines.append("")

    results_sorted = sorted(results, key=lambda x: x["tension"], reverse=True)
    for r in results_sorted:
        if r["status"] == "path broken":
            icon = "🔴"
        elif r["tension"] > 0.6:
            icon = "🔴"
        elif r["tension"] > 0.3:
            icon = "⚠️"
        else:
            icon = "✅"
        lines.append(f"{icon}  {r['name']:<16} tension: {r['tension']:.3f}   {r['status']}")

    lines.append(f"\nOverall gap V(t): {V:.3f}")

    if V > 0.7:
        lines.append("Verdict: Large gap across multiple dimensions — significant effort needed")
    elif V > 0.4:
        lines.append("Verdict: Noticeable gap — focus on the red dimensions first")
    elif V > 0.15:
        lines.append("Verdict: Getting close — keep the momentum")
    else:
        lines.append("Verdict: All dimensions near target — stay consistent")

    lines.append(f"\nTop priority: {results_sorted[0]['name']}")
    lines.append("")
    lines.append("=" * 40)
    lines.append("Reply with your updated progress and Knight will recalculate your tension field.")

    return "\n".join(lines)


def classify_intent(body):
    # L1 Input Layer: route incoming email to the correct processing pipeline
    # Returns one of five intent types; defaults to new_goal if LLM returns unexpected output
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""
Classify this message into exactly one type. Return only the type keyword, nothing else.

Message: {body}

Types:
new_goal  → new long-term goal: applications, exams, fitness, learning, etc.
progress  → progress update: reporting what was done today
question  → information query: weather, facts, knowledge
chat      → casual conversation
urgent    → time-critical situation, deadline imminent
"""
        }]
    )
    result = response.content[0].text.strip()
    valid = {"new_goal", "progress", "question", "chat", "urgent"}
    # Fallback to new_goal if LLM returns unexpected value
    return result if result in valid else "new_goal"


def handle_question(body):
    # Handle information queries using web search
    # Claude decides whether to search or answer directly
    # Future: delegate search to execution layer instead of web_search tool
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system="You are Knight, an AI orchestration assistant developed by Alen Pu. You route tasks to the most suitable models and help users track long-term goals, calculate gaps, and prioritize actions. Unless the user explicitly asks, do not mention the underlying model or Anthropic.",
        max_tokens=500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": body}]
    )
    # Collect all text blocks — tool use splits response into multiple blocks
    texts = []
    for block in response.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    return "\n".join(texts) if texts else ""


def handle_chat(body):
    # Handle casual conversation — no search, no goal analysis
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system="You are Knight, an AI orchestration assistant developed by Alen Pu. You route tasks to the most suitable models and help users track long-term goals, calculate gaps, and prioritize actions. Unless the user explicitly asks, do not mention the underlying model or Anthropic.",
        max_tokens=300,
        messages=[{"role": "user", "content": body}]
    )
    return response.content[0].text


def run_knight(body):
    # Full goal analysis pipeline:
    # 1. Extract goal and deadline from email
    # 2. Get or search requirements (with cache)
    # 3. Extract current status from email
    # 4. Compute tension for each dimension
    # 5. Return results for formatting
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
    # Connect to Gmail via IMAP and fetch unread emails from self
    # Filtered to only process emails sent from the same address (avoids spam/notifications)
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

        # Extract plain text body from potentially multipart email
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
    # Send Knight's analysis back to the user via Gmail SMTP
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


# Main loop: poll inbox every 60 seconds
# No API calls when inbox is empty — zero cost idle
print("Knight email service started...")
print(f"Listening on: {EMAIL_ADDRESS}")
print("Checking for new emails every 60 seconds\n")

while True:
    try:
        emails = check_inbox()
        if emails:
            print(f"Received {len(emails)} new email(s)")
        for e in emails:
            print(f"Processing email from {e['sender']}...")

            # L1: classify intent and route to correct handler
            intent = classify_intent(e["body"])
            print(f"Intent: {intent}")

            if intent in ("new_goal", "progress", "urgent"):
                # Full Knight pipeline: search, extract, compute tension
                goal, V, results, requirements = run_knight(e["body"])
                content = format_results(goal, V, results, requirements)
            elif intent == "question":
                # Direct answer with optional web search
                content = handle_question(e["body"])
            else:
                # Casual reply
                content = handle_chat(e["body"])

            send_reply(e["sender"], e["subject"], content)
            print(f"Replied to {e['sender']}")
    except Exception as ex:
        print(f"Error: {ex}")

    time.sleep(60)
