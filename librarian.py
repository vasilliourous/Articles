#!/usr/bin/env python3
"""
Librarian — Clean Markdown Article Fetcher
Uses Crawl4AI for JS rendering, content extraction, and noise-free Markdown.
Saves articles as readable .md files — no API keys needed.
"""

import os
import re
import hashlib
import unicodedata
import asyncio
import shutil
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from tqdm import tqdm

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BOOKMARK_FILE = "bookmarks.html"
OUTPUT_DIR = "summaries"
FAILED_DIR = "failed"

# How many articles to fetch in parallel (adjust based on your network/CPU)
MAX_CONCURRENT = 5

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(FAILED_DIR).mkdir(exist_ok=True)


# -------------------------------------------------------------------
# Bot / blocked-page detection
# -------------------------------------------------------------------
# Patterns in the RAW markdown that mean "this is a bot wall, not the article"
BOT_BLOCK_PATTERNS: list[str] = [
    r"just a moment",
    r"checking your browser",
    r"ddos protection",
    r"attention required",
    r"enable javascript",
    r"verify you are (human|not a bot)",
    r"security check",
    r"please turn javascript on",
    r"one more step",
    r"performing security verification",
    r"this website uses a security service",
    r"verification successful.*waiting for",
    r"ray id:\s*[0-9a-f]{16}",
    # 403 / forbidden pages returned as "success" by the server
    r"403[-\s]*(forbidden|error)",
    r"access (to this page )?is forbidden",
    r"access (to this page )?is denied",
    r"blocked by the server",
    # DataDome / bot-detection captchas sometimes slip through
    r"datadome.*captcha",
    r"blocked by anti.bot",
]

# Boilerplate / UI chrome to strip from saved markdown
BOILERPLATE_REMOVALS: list[str] = [
    r"Skip to main content",
    r"Join\s+(Tom's Hardware|our|the).*?(Premium|Member|Newsletter)",
    r"Upgrade to.*?Premium",
    r"Become a (member|premium member|subscriber)",
    r"Go beyond the headlines",
    r"Choose how you want to join",
    r"Get started with free access",
    r"Unlock exclusive tools",
    r"Bench Performance Database",
    r"Explore PREMIUM|GO PREMIUM",
    r"Sign up|Subscribe to (our|the)",
    r"newsletter|mailing list",
    r"cookie (notice|policy|consent|settings)",
    r"This website uses cookies",
    r"Accept (All|Recommended|Cookies)",
    r"Reject (All|non.essential)",
    r"Cookie Settings",
    r"Tap to unmute",
    r"If playback doesn't begin shortly",
    r"You're signed out",
    r"Share\s*[·•]\s*Include playlist",
    r"An error occurred while retrieving",
    r"Watch full video",
    r"Up next\s*Live Upcoming",
    r"CancelPlay Now",
    r"Back\s+\[.*?\]\(.*?\)",
    r"Search\s+\[.*?\]\(.*?\)",
    r"\[\s*\]\(.*?\)\s*$",  # orphaned empty image references
]

# Patterns in crawl4ai's error_message that indicate a bot block
BOT_BLOCK_ERROR_PATTERNS: list[str] = [
    r"blocked by anti.bot",
    r"datadome",
    r"captcha",
    r"cloudflare",
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def is_bot_blocked(content: str) -> bool:
    """Return True if *content* looks like a bot/security challenge."""
    if not content:
        return False
    lower = content.lower()
    return any(re.search(p, lower) for p in BOT_BLOCK_PATTERNS)


def is_bot_block_error(error_msg: Optional[str]) -> bool:
    """Return True if the crawl4ai error message indicates a bot block."""
    if not error_msg:
        return False
    lower = error_msg.lower()
    return any(re.search(p, lower) for p in BOT_BLOCK_ERROR_PATTERNS)


def clean_markdown(md: str, url: str) -> str:
    """Strip known boilerplate / UI chrome from the markdown."""
    if not md:
        return md

    lines = md.split("\n")
    cleaned: list[str] = []
    for line in lines:
        skip = False
        for pat in BOILERPLATE_REMOVALS:
            if re.search(pat, line, re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    text = "\n".join(l.rstrip() for l in text.split("\n"))
    return text.strip()


def slugify_filename(title: str, url_hash: str, suffix: str = ".md") -> str:
    """Convert a title to a filesystem-safe slug, prefixed with a short hash."""
    slug = unicodedata.normalize("NFKD", title)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-").lower()
    slug = slug[:80]
    if not slug:
        slug = "untitled"
    return f"{url_hash[:8]}-{slug}{suffix}"


def build_article_md(title: str, url: str, content_md: str) -> str:
    """Wrap the clean Markdown with article metadata header."""
    return f"""---
title: {title}
source: {url}
---

{content_md}"""


def get_done_hashes() -> set[str]:
    """Build set of URL-MD5 hashes already saved as .md files.

    Reads the ``source:`` frontmatter field from every existing .md file.
    Falls back to the old filename-as-hash convention for legacy files.
    """
    done: set[str] = set()
    for f in Path(OUTPUT_DIR).glob("*.md"):
        head = f.read_text(encoding="utf-8", errors="ignore")[:2000]
        m = re.search(r"^source:\s*(.+)$", head, re.MULTILINE)
        if m:
            url = m.group(1).strip()
            done.add(hashlib.md5(url.encode()).hexdigest())
            continue
        # Legacy files that are just the bare MD5 hash
        stem = f.stem
        if re.match(r"^[0-9a-f]{32}$", stem):
            done.add(stem)
    return done


# -------------------------------------------------------------------
# Browser / crawl configs
# -------------------------------------------------------------------
def make_browser_config() -> BrowserConfig:
    """Build the Crawl4AI browser config with stealth enhancements."""
    return BrowserConfig(
        headless=True,
        enable_stealth=True,
        use_managed_browser=True,
        light_mode=False,
        extra_args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ],
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        ignore_https_errors=True,
    )


def make_crawl_config(cache_mode: CacheMode = CacheMode.DISABLED) -> CrawlerRunConfig:
    """Build a crawl config with tuned filtering.

    Uses ``CacheMode.DISABLED`` by default so that previously cached
    403 / garbage pages don't poison retries.
    """
    prune_filter = PruningContentFilter(
        threshold=0.25,
        threshold_type="dynamic",
        min_word_threshold=3,
    )
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    return CrawlerRunConfig(
        cache_mode=cache_mode,
        markdown_generator=md_generator,
        word_count_threshold=5,
        excluded_tags=[
            "nav", "footer", "header", "form", "aside",
            "noscript", "script", "style",
        ],
        exclude_external_links=True,
        exclude_social_media_links=True,
        remove_consent_popups=True,
        process_iframes=False,
    )


# -------------------------------------------------------------------
# Bookmark parsing
# -------------------------------------------------------------------
def parse_bookmarks(html_file: str) -> list[dict]:
    """Parse bookmarks HTML, deduplicating by URL."""
    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    seen = set()
    links = []
    for tag in soup.find_all("a"):
        url = tag.get("href")
        title = tag.get_text(strip=True)
        if url and url.startswith("http") and url not in seen:
            seen.add(url)
            links.append({"url": url, "title": title or url})
    return links


# -------------------------------------------------------------------
# Single-article classification + save
# -------------------------------------------------------------------
async def _fetch_and_save(
    crawler: AsyncWebCrawler,
    url: str,
    title: str,
    url_hash: str,
    crawl_config: CrawlerRunConfig,
    used_slugs: set[str],
    pbar: tqdm,
    success_count: list[int],
    fail_count: list[int],
    block_count: list[int],
    retry: bool = False,
) -> None:
    """Fetch one URL and save or classify the result.

    Uses mutable 1-element lists as counters so the function can increment
    them from inside the async pipeline.
    """
    result = await crawler.arun(url=url, config=crawl_config)

    # --- Determine the outcome -------------------------------------------------
    reason: Optional[str] = None  # set => failure; None => success

    if not result.success:
        error_msg = result.error_message or "unknown error"
        if is_bot_block_error(error_msg):
            reason = f"blocked — {error_msg}"
        else:
            reason = f"error — {error_msg}"
    else:
        md = result.markdown
        content = (
            md.fit_markdown
            if md.fit_markdown and len(md.fit_markdown) > 100
            else md.raw_markdown
        )

        if not content or len(content) <= 100:
            reason = f"content too short ({len(content) if content else 0} chars)"
        elif is_bot_blocked(content):
            reason = "blocked — anti-bot/security page detected"

    # --- Act on the outcome ----------------------------------------------------
    if reason is not None:
        if "blocked" in reason:
            tqdm.write(f"  🛡️  '{title}' — {reason}")
            block_count[0] += 1
        else:
            tqdm.write(f"  ⚠️  '{title}' — {reason}")
            fail_count[0] += 1

        with open(Path(FAILED_DIR) / f"{url_hash}.txt", "w") as f:
            f.write(f"{url}\n# {reason}")

        pbar.update(1)
        pbar.set_postfix(ok=success_count[0], fail=fail_count[0], blocked=block_count[0])
        return

    # --- Success path ----------------------------------------------------------
    cleaned = clean_markdown(content, url)
    if not cleaned:
        cleaned = content  # keep original rather than save nothing

    article_md = build_article_md(title, url, cleaned)
    fname = slugify_filename(title, url_hash)
    while fname in used_slugs:
        stem = Path(fname).stem
        fname = f"{stem}-dup.md"
    used_slugs.add(fname)

    with open(Path(OUTPUT_DIR) / fname, "w", encoding="utf-8") as f:
        f.write(article_md)

    success_count[0] += 1
    pbar.update(1)
    pbar.set_postfix(ok=success_count[0], fail=fail_count[0], blocked=block_count[0])


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
async def process_articles():
    """Main async pipeline: fetch articles and save as clean Markdown."""
    print("Parsing bookmarks ...")
    bookmarks = parse_bookmarks(BOOKMARK_FILE)
    print(f"Found {len(bookmarks)} unique URLs.")

    done_hashes = get_done_hashes()
    todo = [bm for bm in bookmarks
            if hashlib.md5(bm["url"].encode()).hexdigest() not in done_hashes]

    print(f"Already processed: {len(bookmarks) - len(todo)}"
          f"  |  Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    # Mutable counters for the async pipeline
    success_count = [0]
    fail_count = [0]
    block_count = [0]
    used_slugs: set[str] = set()

    browser_config = make_browser_config()
    crawl_config = make_crawl_config(cache_mode=CacheMode.DISABLED)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        with tqdm(total=len(todo), desc="Fetching articles", unit="article") as pbar:
            # Use asyncio.gather with a semaphore for concurrency control
            sem = asyncio.Semaphore(MAX_CONCURRENT)

            async def throttled_fetch(bm: dict) -> None:
                async with sem:
                    url_hash = hashlib.md5(bm["url"].encode()).hexdigest()
                    await _fetch_and_save(
                        crawler,
                        bm["url"],
                        bm["title"],
                        url_hash,
                        crawl_config,
                        used_slugs,
                        pbar,
                        success_count,
                        fail_count,
                        block_count,
                    )

            await asyncio.gather(*[throttled_fetch(bm) for bm in todo])

    # Clean up stray .obsidian that leaked into the output directory
    obsidian_dir = Path(OUTPUT_DIR) / ".obsidian"
    if obsidian_dir.exists():
        shutil.rmtree(obsidian_dir)
        print(f"   🧹 Removed stray .obsidian from {OUTPUT_DIR}/")

    print(f"\n✅ Finished!")
    print(f"   Success:   {success_count[0]} new articles")
    print(f"   Blocked:   {block_count[0]} (anti-bot protection)")
    print(f"   Failed:    {fail_count[0]}")
    print(f"   Total .md on disk: {len(list(Path(OUTPUT_DIR).glob('*.md')))} files")
    print(f"   Old .json kept: {len(list(Path(OUTPUT_DIR).glob('*.json')))} files")
    if fail_count[0] or block_count[0]:
        print("   Re‑run the script to retry failed ones automatically.")


def main():
    try:
        asyncio.run(process_articles())
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted — progress saved. Re-run to continue where you left off.")


if __name__ == "__main__":
    main()
