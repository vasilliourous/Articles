#!/usr/bin/env python3
"""
Librarian — Personalised Article Summariser (Scrapling + Detailed Output)
Fetches pages with a stealth browser, then uses DeepSeek to create
rich, personalised summaries that respect your interests.
"""

import os
import json
import hashlib
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm
from dotenv import load_dotenv
from scrapling import StealthyFetcher

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY. Create a .env file with that variable.")

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

BOOKMARK_FILE = "bookmarks.html"
OUTPUT_DIR = "summaries"
FAILED_DIR = "failed"
TXT_CACHE_DIR = "clean_texts"

SLEEP_BETWEEN_CALLS = 2
REQUEST_TIMEOUT = 120
FETCH_TIMEOUT = 30000          # milliseconds for Scrapling

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(FAILED_DIR).mkdir(exist_ok=True)
Path(TXT_CACHE_DIR).mkdir(exist_ok=True)

# Load personal needs
NEEDS_FILE = "my_needs.md"
if not Path(NEEDS_FILE).exists():
    raise FileNotFoundError(
        f"'{NEEDS_FILE}' not found. Create it and write your interests/needs there."
    )
with open(NEEDS_FILE, "r", encoding="utf-8") as f:
    USER_NEEDS = f.read().strip()

# -------------------------------------------------------------------
# Detailed personalised system prompt
# -------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a personal research librarian. Below is a description of the user's interests, needs, and requirements (which may be ambiguous or loosely defined). Use it as a lens for summarising the article.

<user_needs>
{USER_NEEDS}
</user_needs>

You will receive an article title and its full text. Your task is to produce a detailed, informative JSON summary. Do not oversimplify; instead, write thorough explanations that preserve the depth of the original. Fields should be descriptive and context‑rich, while still structured.

Use this exact JSON structure:

{{
  "article_title": "title",
  "core_thesis": "one paragraph summarising the main argument, including any nuance or context",
  "detailed_summary": "2-3 paragraphs that give a complete overview of the article. Include key events, reasoning, data, and any critical context so that a reader can understand the full picture without reading the original.",
  "entities": [
    {{"name": "entity name", "type": "person/organization/location/event", "role": "brief description of their relevance"}}
  ],
  "key_data_points": [
    "A detailed description of each important fact, statistic, finding, or statement. Write each as a complete sentence (or two) that explains the data and its significance. Mark relevant items with 🟢 and explain why they matter to you.",
    "Example: 'The vulnerability CVE-2025-1234 allows remote code execution via crafted HTTP headers (🟢 relevant to your Red Teaming: can be weaponised for initial access).'"
  ],
  "methodology_steps": [
    "For how‑to, DevOps, or research articles: write each step as a clear, descriptive instruction. Include commands, configurations, or code where applicable. If DIY‑related, explain the reasoning behind the step."
  ],
  "conclusions": [
    "Each conclusion should be a well‑rounded statement, with supporting evidence from the article. Link it to the user's interests where possible."
  ],
  "personal_relevance_summary": "A thorough 3‑5 sentence paragraph explaining how this article connects to the user's described needs, what the user can take away from it, and why it's worth their attention. Be specific.",
  "extra_advice": "Optional. If you see a clear, actionable next step, idea, or opportunity (matching the user's ambition), provide a concise but concrete suggestion. Otherwise omit this field entirely."
}}

Important behavioural rules:
- The user is **ambitious and appreciates ideas/advice**; do not be afraid to draw connections or suggest practical applications.
- For **Cybersecurity (Red Teaming)**: always describe vulnerabilities through an offensive‑security lens – how can they be exploited, what tools or techniques, etc.
- For **Homelab / DevOps**: extract technical details, architecture, and configuration steps in full, not just summaries. Include pro‑tips if present.
- For **app/tool recommendations**: if many tools are mentioned, keep each tool description to 1‑2 sentences. If only one tool is the article's focus, you may elaborate freely.
- For **Reddit discussions**: synthesise the overall consensus, majority opinion, or the most insightful conclusion, not a list of random comments.
- The user is a **student** – highlight study techniques, productivity hacks, or psychological insights with practical application.

Write naturally and informatively. The JSON structure is the container, but the content inside should read like an expert summary, not a bullet‑point telegraph.

Return ONLY the JSON object, with no extra text before or after it. If you want to provide extra advice, include it as the optional "extra_advice" field inside the JSON (not outside). Make the JSON valid.
"""

# -------------------------------------------------------------------
# Phase 1 – Bookmark parsing & text extraction (Scrapling)
# -------------------------------------------------------------------
def parse_bookmarks(html_file: str) -> list[dict]:
    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    links = []
    for tag in soup.find_all("a"):
        url = tag.get("href")
        title = tag.get_text(strip=True)
        if url and url.startswith("http"):
            links.append({"url": url, "title": title})
    return links

def fetch_and_clean(url: str) -> str | None:
    """
    Fetch a page with Scrapling (stealth browser), wait for JavaScript to
    finish rendering, then extract the full visible text. Finally pass the
    HTML through Trafilatura for further cleaning.
    """
    try:
        # network_idle=True – waits until no network activity for 500ms
        page = StealthyFetcher.fetch(url, timeout=FETCH_TIMEOUT, network_idle=True)
        if page is None:
            print(f"    Scrapling returned None for: {url}")
            return None

        # .text is often empty on JS‑heavy pages; .get_all_text() works universally
        html = page.get_all_text()
        if not html or len(html) < 200:
            print(f"    Content too short for: {url}")
            return None

        # Trafilatura still expects HTML, but it can also handle plain text.
        # If the page is mostly text already, this just cleans up whitespace.
        text = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
        )
        # Fallback: if Trafilatura couldn't parse it, return the raw text
        if not text or len(text) < 100:
            text = html
        return text
    except Exception as e:
        print(f"    Scrapling error for {url}: {e}")
        return None

def get_clean_text(bookmark: dict) -> str | None:
    safe_name = hashlib.md5(bookmark["url"].encode()).hexdigest()
    txt_path = Path(TXT_CACHE_DIR) / f"{safe_name}.txt"
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            cached = f.read()
        if len(cached) > 50:
            return cached
    text = fetch_and_clean(bookmark["url"])
    if text and len(text) > 100:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    return None

# -------------------------------------------------------------------
# Phase 2 – Personalised compression via DeepSeek
# -------------------------------------------------------------------
def compress_article(article_text: str, title: str) -> dict:
    user_message = f"Title: {title}\n\nArticle text:\n{article_text}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Extract JSON (the prompt now asks for pure JSON, no outside text)
    if "```" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]

    result = json.loads(content)
    return result

# -------------------------------------------------------------------
# Main sequential loop
# -------------------------------------------------------------------
def main():
    print("Parsing bookmarks ...")
    bookmarks = parse_bookmarks(BOOKMARK_FILE)
    print(f"Found {len(bookmarks)} URLs.")

    processed_hashes = {f.stem for f in Path(OUTPUT_DIR).glob("*.json")}
    todo = []
    for bm in bookmarks:
        url_hash = hashlib.md5(bm["url"].encode()).hexdigest()
        if url_hash not in processed_hashes:
            todo.append(bm)

    print(f"Already processed: {len(bookmarks) - len(todo)}. Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    for bookmark in tqdm(todo, desc="Processing articles", unit="article"):
        title = bookmark["title"]
        url = bookmark["url"]
        url_hash = hashlib.md5(url.encode()).hexdigest()

        clean_text = get_clean_text(bookmark)
        if not clean_text:
            tqdm.write(f"⚠️  Skipping '{title}' — no usable text extracted.")
            with open(Path(FAILED_DIR) / f"{url_hash}.txt", "w") as f:
                f.write(url)
            continue

        try:
            result = compress_article(clean_text, title)
            result["source_url"] = url
            out_path = Path(OUTPUT_DIR) / f"{url_hash}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"❌ Failed '{title}': {e}")
            with open(Path(FAILED_DIR) / f"{url_hash}.txt", "w") as f:
                f.write(url)
            time.sleep(10)
            continue

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n✅ Finished!")
    print(f"   Success: {len(list(Path(OUTPUT_DIR).glob('*.json')))} detailed summaries")
    print(f"   Failed:  {len(list(Path(FAILED_DIR).glob('*.txt')))} articles (see 'failed' folder)")
    print("   Re‑run the script to retry failed ones automatically.")

if __name__ == "__main__":
    main()
