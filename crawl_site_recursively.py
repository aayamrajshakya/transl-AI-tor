"""
REFERENCE: https://github.com/coleam00/ottomator-agents/blob/main/crawl4AI-agent-v2/crawl4AI-examples/5-crawl_site_recursively.py
Recursively crawls a site starting from a root URL, using Crawl4AI's arun_many and a memory-adaptive dispatcher.
At each depth, all internal links are discovered and crawled in parallel, up to a specified depth, with deduplication.
"""

import asyncio
from urllib.parse import urldefrag, urlparse
import os
import re
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)


def create_filename_from_url(url: str) -> str:
    """
    Creates a filename from URL path after the domain

    Args:
        url (str): The full URL

    Returns:
        str: Filename for the content
    """
    parsed = urlparse(url)
    path = parsed.path

    if path.startswith("/"):
        path = path[1:]

    if not path or path == "":
        filename = "index"
    else:
        filename = path.replace("/", "_").replace("\\", "_")
        filename = re.sub(r"[^\w\-_.]", "_", filename)

    if not filename.endswith(".md"):
        filename += ".md"

    return filename


def save_content_to_file(content: str, url: str, base_folder: str):
    """
    Saves the scraped content to a file in the specified folder structure.

    Args:
        content (str): The markdown content to save
        url (str): The original URL
        base_folder (str): The base folder name (e.g., 'www.youxel.com')
    """
    try:
        os.makedirs(base_folder, exist_ok=True)

        filename = create_filename_from_url(url)
        filepath = os.path.join(base_folder, filename)

        try:
            with open(filepath, "x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError:
            print(f"File already exists, skipping: {filepath}")
            return

        print(f"Saved content to: {filepath}")

    except Exception as e:
        print(f"Error saving content for {url}: {e}")


def extract_domain_from_url(url: str) -> str:
    """
    Extracts the domain name from the URL to use as folder name.

    Args:
        url (str): The URL

    Returns:
        str: Domain name to use as folder name
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    return domain


async def crawl_recursive_batch(start_urls, max_depth=3, max_concurrent=10):
    # Extract base folder name from the first start URL
    base_folder = (
        extract_domain_from_url(start_urls[0]) if start_urls else "crawled_content"
    )
    print(f"Saving content to folder: {base_folder}")

    # Reference: https://docs.crawl4ai.com/api/parameters/
    ## Controlling the browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        java_script_enabled=False,  # No JS overhead, only static content
        text_mode=True,  # Tries to disable images/other heavy content for speed
    )

    # Reference: https://github.com/unclecode/crawl4ai/issues/181#issuecomment-2425694758
    ## Controlling each crawl
    run_config = CrawlerRunConfig(
        word_count_threshold=50,  # Skips text blocks below X words
        remove_forms=True,
        only_text=True,
        cache_mode=CacheMode.BYPASS,
        wait_for_images=False,
        remove_overlay_elements=True,  # Removes potential modals/popups blocking the main content
        stream=False,
        exclude_all_images=True,
        exclude_social_media_links=True,
        excluded_tags=["header", "footer", "nav"],
        # wait_until="domcontentloaded"   # Condition for navigation to "complete"
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,  # Pause if memory exceeds this
        check_interval=1.0,  # How often to check memory
        max_session_permit=max_concurrent,  # Max parallel browser sessions
    )

    SKIP_EXTENSIONS = {
        ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
        ".zip", ".tar", ".gz", ".mp4", ".mp3", ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx", ".csv"
    }

    def is_crawlable(url: str) -> bool:
        path = urlparse(url).path.lower()
        return not any(path.endswith(ext) for ext in SKIP_EXTENSIONS)

    # Track visited URLs to prevent revisiting and infinite loops (ignoring fragments)
    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = {normalize_url(u) for u in start_urls}

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            print(f"\n=== Crawling Depth {depth+1} ===")
            urls_to_crawl = list(current_urls - visited)

            if not urls_to_crawl:
                break

            # Batch-crawl all URLs at this depth in parallel
            results = await crawler.arun_many(
                urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
            )

            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)
                if result.success:
                    print(
                        f"[OK] {result.url} | Markdown: {len(result.markdown) if result.markdown else 0} chars"
                    )
                    # Save the content to file
                    save_content_to_file(
                        result.markdown.raw_markdown, result.url, base_folder
                    )
                    # Collect all new `internal` links for the next depth
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])

                        if next_url not in visited and is_crawlable(next_url):
                            next_level_urls.add(next_url)
                else:
                    print(f"[ERROR] {result.url}: {result.error_message}")

            # Move to the next set of URLs for the next recursion depth
            current_urls = next_level_urls


if __name__ == "__main__":
    asyncio.run(
        crawl_recursive_batch(
            ["https://gouvernement.lu/lb"],
            max_depth=5,
            max_concurrent=20,
        )
    )
    