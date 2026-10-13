"""
download_official_documents.py
==============================
Fetches and saves the AUTHENTIC, OFFICIAL engineering post-mortems:
1. CrowdStrike: Official 12-Page External Root Cause Analysis PDF directly from CrowdStrike.
2. Cloudflare: Official post-mortem directly from the Cloudflare Blog.
3. GitHub: Official post-mortem directly from the GitHub Engineering Blog.
4. AWS: Official post-event summary directly from AWS message archive.
"""

import os
import re
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "postmortems"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_crowdstrike_pdf():
    url = "https://www.crowdstrike.com/wp-content/uploads/2024/08/Channel-File-291-Incident-Root-Cause-Analysis-08.06.2024.pdf"
    dest = OUTPUT_DIR / "crowdstrike_outage_2024.pdf"
    print(f"\n[1/4] Downloading Official CrowdStrike RCA PDF from:")
    print(f"      {url}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        size_kb = dest.stat().st_size / 1024
        print(f"      ✅ SAVED: {dest.name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"      ❌ Download error: {e}")


def clean_html_to_markdown(html_text: str, title: str, source_url: str) -> str:
    """Extracts text content and converts paragraphs and headers to clean Markdown."""
    # Strip script and style blocks
    cleaned = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', cleaned, flags=re.IGNORECASE)
    
    # Convert headings
    cleaned = re.sub(r'<h1[^>]*>(.*?)<\/h1>', r'\n# \1\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<h2[^>]*>(.*?)<\/h2>', r'\n## \1\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<h3[^>]*>(.*?)<\/h3>', r'\n### \1\n', cleaned, flags=re.IGNORECASE)
    
    # Convert paragraphs and line breaks
    cleaned = re.sub(r'<p[^>]*>(.*?)<\/p>', r'\n\1\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<li[^>]*>(.*?)<\/li>', r'\n- \1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<br\s*\/?>', '\n', cleaned, flags=re.IGNORECASE)
    
    # Strip all remaining tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Unescape HTML entities
    cleaned = cleaned.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
    
    # Condense consecutive blank lines
    lines = [line.strip() for line in cleaned.splitlines()]
    meaningful = [l for l in lines if l]
    body = "\n\n".join(meaningful)

    return f"# {title}\n\n**Official Source:** {source_url}\n\n---\n\n{body}"


def fetch_official_markdown(url: str, dest_filename: str, title: str, step_num: int):
    dest = OUTPUT_DIR / dest_filename
    print(f"\n[{step_num}/4] Fetching Official Post-Mortem from:")
    print(f"      {url}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
        md_content = clean_html_to_markdown(html, title, url)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(md_content)
        size_kb = dest.stat().st_size / 1024
        print(f"      ✅ SAVED: {dest.name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"      ❌ Fetch error: {e}")


def main():
    print("=" * 75)
    print(" 📥 Downloading Official, Authentic Post-Mortem Documents")
    print("=" * 75)

    download_crowdstrike_pdf()

    fetch_official_markdown(
        url="https://blog.cloudflare.com/how-verizon-and-a-bgp-optimizer-knocked-large-parts-of-the-internet-offline-today/",
        dest_filename="cloudflare_bgp_leak.md",
        title="Official Cloudflare Post-Mortem: How Verizon and a BGP Optimizer Knocked Large Parts of the Internet Offline",
        step_num=2
    )

    fetch_official_markdown(
        url="https://github.blog/news-insights/company-news/oct21-post-incident-analysis/",
        dest_filename="github_mysql_failover.md",
        title="Official GitHub Post-Mortem: October 21 Post-Incident Analysis",
        step_num=3
    )

    fetch_official_markdown(
        url="https://aws.amazon.com/message/11201/",
        dest_filename="aws_kinesis_outage.md",
        title="Official AWS Post-Event Summary: Amazon Kinesis Event in Northern Virginia (US-EAST-1)",
        step_num=4
    )

    print("\n" + "=" * 75)
    print(" ✅ Complete! Official documents downloaded to data/postmortems/")
    print("=" * 75)


if __name__ == "__main__":
    main()
