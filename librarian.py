#!/usr/bin/env python3
"""
Librarian — Clean Markdown Article Fetcher
Uses Crawl4AI for JS rendering, content extraction, and noise-free Markdown.
Saves articles as readable .md files — no API keys needed.
"""

import os
import hashlib
import asyncio
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
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
# Markdown generation helpers
# -------------------------------------------------------------------
def build_article_md(title: str, url: str, content_md: str) -> str:
    """Wrap the clean Markdown with article metadata header."""
    return f"""---
title: {title}
source: {url}
---

{content_md}"""


def make_crawl_config() -> CrawlerRunConfig:
    """Build the Crawl4AI config with content filtering."""
    prune_filter = PruningContentFilter(
        threshold=0.45,
        threshold_type="fixed",
        min_word_threshold=5,
    )
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    return CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=md_generator,
        word_count_threshold=10,
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
# Article processing
# -------------------------------------------------------------------
async def process_articles():
    """Main async pipeline: fetch articles and save as clean Markdown."""
    print("Parsing bookmarks ...")
    bookmarks = parse_bookmarks(BOOKMARK_FILE)
    print(f"Found {len(bookmarks)} unique URLs.")

    # Determine which articles already have clean .md files (skip those)
    processed_md = {f.stem for f in Path(OUTPUT_DIR).glob("*.md")}
    # Skip previously failed URLs so we don't hammer broken links
    failed = {f.stem for f in Path(FAILED_DIR).glob("*.txt")}
    # Old .json summaries are NOT skipped — they'll be re-fetched as .md
    already_done = processed_md | failed

    todo = []
    for bm in bookmarks:
        url_hash = hashlib.md5(bm["url"].encode()).hexdigest()
        if url_hash not in already_done:
            todo.append(bm)

    print(
        f"Already processed: {len(bookmarks) - len(todo)}"
        f"  |  Remaining: {len(todo)}"
    )
    if not todo:
        print("Nothing to do.")
        return

    config = make_crawl_config()
    success_count = 0
    fail_count = 0

    async with AsyncWebCrawler() as crawler:
        # Build a list of (url, title) tuples for the todo items
        url_title_pairs = [(bm["url"], bm["title"]) for bm in todo]
        url_hash_lookup = {
            hashlib.md5(url.encode()).hexdigest(): (url, title)
            for url, title in url_title_pairs
        }

        with tqdm(total=len(todo), desc="Fetching articles", unit="article") as pbar:
            results = await crawler.arun_many(
                [url for url, _ in url_title_pairs],
                config=config,
                stream=True,
            )
            for result in results:
                url_hash = hashlib.md5(result.url.encode()).hexdigest()
                # Look up original title from bookmarks; fall back to page's own title
                _, title = url_hash_lookup.get(url_hash, (result.url, result.metadata.get("title", "Untitled") if result.metadata else "Untitled"))

                if result.success:
                    md = result.markdown
                    # Prefer fit_markdown (noise-filtered), fall back to raw
                    content = (
                        md.fit_markdown
                        if md.fit_markdown and len(md.fit_markdown) > 100
                        else md.raw_markdown
                    )

                    if content and len(content) > 100:
                        article_md = build_article_md(title, result.url, content)
                        out_path = Path(OUTPUT_DIR) / f"{url_hash}.md"
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(article_md)
                        success_count += 1
                    else:
                        tqdm.write(
                            f"  ⚠️  '{title}' — content too short "
                            f"({len(content) if content else 0} chars)"
                        )
                        with open(Path(FAILED_DIR) / f"{url_hash}.txt", "w") as f:
                            f.write(result.url)
                        fail_count += 1
                else:
                    tqdm.write(
                        f"  ❌ '{title}' — {result.error_message}"
                    )
                    with open(Path(FAILED_DIR) / f"{url_hash}.txt", "w") as f:
                        f.write(result.url)
                    fail_count += 1

                pbar.update(1)
                pbar.set_postfix(ok=success_count, fail=fail_count)

    print(f"\n✅ Finished!")
    print(f"   Success: {success_count} new articles")
    print(f"   Failed:  {fail_count}")
    print(f"   Total .md on disk: {len(list(Path(OUTPUT_DIR).glob('*.md')))} files")
    print(f"   Old .json kept: {len(list(Path(OUTPUT_DIR).glob('*.json')))} files")
    if fail_count:
        print("   Re‑run the script to retry failed ones automatically.")


def main():
    asyncio.run(process_articles())


if __name__ == "__main__":
    main()
