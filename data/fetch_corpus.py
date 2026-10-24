"""
fetch_corpus.py
===============
Phase 1 of the IncidentLens plan.

This script builds the incident corpus. It does three tasks:

1. It reads a manifest of authentic, public post-mortems.
   Each manifest entry holds: the URL, the organization, the date, the system,
   and the failure category.
2. It downloads each source and extracts the article text. It removes the
   navigation, the scripts, the styles and the footers.
3. It writes one Markdown file for each source. It writes a metadata sidecar
   in JSON. It writes a report of the successes and the failures.

Why this file exists
--------------------
The first corpus had 4 documents. Retrieval scored a perfect 1.000 on that
corpus. A perfect score cannot show an improvement, so no retrieval technique
could be measured. A larger corpus removes that limit.

Design rules
------------
* One manifest entry makes one document.
* A domain-specific selector list gives a clean article body.
* An extraction below the word floor is a FAILURE, not a document. A short
  extraction is usually a JavaScript page or a navigation menu.
* The script never raises on a single bad source. It records the failure and
  continues. Therefore one dead URL cannot stop the build.

Usage
-----
    python fetch_corpus.py                # download everything that is missing
    python fetch_corpus.py --force        # download again, overwrite
    python fetch_corpus.py --only aws     # only the sources whose id starts with "aws"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../data
PROJECT = HERE.parent                             # project root
OUT_DIR = HERE / "postmortems"
META_PATH = HERE / "corpus_manifest.json"
REPORT_PATH = HERE / "corpus_report.md"

# --------------------------------------------------------------------------
# Download settings
# --------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 40
RETRIES = 2
DELAY_BETWEEN = 0.4          # seconds. Be polite to the source servers.
WORD_FLOOR = 250             # an extraction below this value is a failure.

# Tags that never hold article text.
DROP_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "form",
    "noscript", "iframe", "svg", "button", "figure", "picture", "template",
]

# Selector order matters. The first selector that gives enough text wins.
# A general source uses these. A specific source can override them.
DEFAULT_SELECTORS = [
    ".entry-content",
    "[role='main']",
    ".prose",
    "article",
    "main",
    "body",
]


# --------------------------------------------------------------------------
# Source definition
# --------------------------------------------------------------------------
@dataclass
class Source:
    """One public post-mortem."""

    sid: str                 # short stable id, for example "aws-eu-west-2011"
    url: str
    org: str                 # the organization that wrote the report
    published: str           # ISO date of the incident or the report
    system: str              # the system that failed, for example "network"
    failure: str             # the failure category, for example "power"
    title: str
    selectors: list = field(default_factory=list)   # empty means DEFAULT_SELECTORS


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------
SOURCES: list = [
    # ---------------- AWS post-event summaries ----------------
    Source("aws-eu-west-power-2011", "https://aws.amazon.com/message/2329B7/",
           "Amazon Web Services", "2011-08-07", "EC2/EBS/RDS", "power",
           "Amazon EC2, EBS and RDS Service Event in the EU West Region"),
    Source("aws-us-east-power-2012", "https://aws.amazon.com/message/67457/",
           "Amazon Web Services", "2012-06-29", "EC2/EBS/RDS", "power",
           "Amazon EC2, EBS and RDS Service Event in the US East Region"),
    Source("aws-sydney-power-2012", "https://aws.amazon.com/message/4372T8/",
           "Amazon Web Services", "2012-06-04", "EC2/EBS", "power",
           "Amazon EC2 and EBS Service Event in the Sydney Region"),
    Source("aws-sa-east-power-2012", "https://aws.amazon.com/message/656481/",
           "Amazon Web Services", "2012-12-21", "EC2/EBS", "power",
           "Amazon EC2 and EBS Service Event in the Sao Paulo Region"),
    Source("aws-tokyo-cooling-2019", "https://aws.amazon.com/message/56489/",
           "Amazon Web Services", "2019-08-23", "EC2/EBS", "cooling",
           "Amazon EC2 and EBS Service Event in the Tokyo Region"),
    Source("aws-seoul-dns-2025", "https://aws.amazon.com/message/74876-2/",
           "Amazon Web Services", "2025-01-01", "Route 53/EC2 DNS", "config",
           "EC2 DNS Resolution Event in the Seoul Region"),
    Source("aws-us-east-kinesis", "https://aws.amazon.com/message/11201/",
           "Amazon Web Services", "2020-11-25", "Kinesis", "capacity",
           "Amazon Kinesis Event in the Northern Virginia Region"),

    # ---------------- Cloudflare ----------------
    Source("cf-2025-11-18", "https://blog.cloudflare.com/18-november-2025-outage/",
           "Cloudflare", "2025-11-18", "core proxy", "config",
           "Cloudflare outage on 18 November 2025"),
    Source("cf-2025-12-05", "https://blog.cloudflare.com/5-december-2025-outage/",
           "Cloudflare", "2025-12-05", "WAF rulesets", "config",
           "Cloudflare outage on 5 December 2025"),
    Source("cf-2026-02-20", "https://blog.cloudflare.com/cloudflare-outage-february-20-2026/",
           "Cloudflare", "2026-02-20", "ByoIP addressing", "software",
           "Cloudflare outage on 20 February 2026"),
    Source("cf-2025-02-06", "https://blog.cloudflare.com/cloudflare-incident-on-february-6-2025/",
           "Cloudflare", "2025-02-06", "R2 Gateway", "human-error",
           "Cloudflare incident on 6 February 2025"),
    Source("cf-2025-03-21", "https://blog.cloudflare.com/cloudflare-incident-march-21-2025/",
           "Cloudflare", "2025-03-21", "R2 Gateway", "human-error",
           "Cloudflare incident on 21 March 2025"),
    Source("cf-2024-06-20", "https://blog.cloudflare.com/cloudflare-incident-on-june-20-2024/",
           "Cloudflare", "2024-06-20", "DDoS mitigation", "software",
           "Cloudflare incident on 20 June 2024"),
    Source("cf-2023-10-30", "https://blog.cloudflare.com/cloudflare-incident-on-october-30-2023/",
           "Cloudflare", "2023-10-30", "Workers KV", "deploy",
           "Cloudflare incident on 30 October 2023"),
    Source("cf-dashboard-2020", "https://blog.cloudflare.com/deep-dive-into-cloudflares-sept-12-dashboard-and-api-outage/",
           "Cloudflare", "2020-09-12", "dashboard API", "software",
           "Deep dive into the Cloudflare dashboard and API outage"),
    Source("cf-2026-01-22-route-leak", "https://blog.cloudflare.com/route-leak-incident-january-22-2026/",
           "Cloudflare", "2026-01-22", "BGP routing", "config",
           "Cloudflare route leak incident on 22 January 2026"),
    Source("cf-2023-01-24", "https://blog.cloudflare.com/cloudflare-incident-on-january-24th-2023/",
           "Cloudflare", "2023-01-24", "service tokens", "deploy",
           "Cloudflare incident on 24 January 2023"),
    Source("cf-2023-10-04-dns", "https://blog.cloudflare.com/1-1-1-1-lookup-failures-on-october-4th-2023/",
           "Cloudflare", "2023-10-04", "1.1.1.1 DNS", "software",
           "1.1.1.1 DNS lookup failures on 4 October 2023"),

    # ---------------- Meta ----------------
    Source("meta-2021-10-04", "https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/",
           "Meta", "2021-10-04", "backbone network", "config",
           "More details about the October 4 outage"),

    # ---------------- GitHub ----------------
    Source("github-2018-10-21", "https://github.blog/news-insights/company-news/oct21-post-incident-analysis/",
           "GitHub", "2018-10-21", "MySQL", "network",
           "October 21 post-incident analysis"),
    Source("github-2014-01-16-dns", "https://github.blog/news-insights/the-library/dns-outage-post-mortem/",
           "GitHub", "2014-01-16", "DNS", "config",
           "DNS outage post-mortem"),
    Source("github-2024-08", "https://github.blog/news-insights/company-news/github-availability-report-august-2024/",
           "GitHub", "2024-08-01", "database routing", "config",
           "GitHub availability report August 2024"),
    Source("github-2026-02", "https://github.blog/news-insights/company-news/github-availability-report-february-2026/",
           "GitHub", "2026-02-01", "VM metadata", "config",
           "GitHub availability report February 2026"),

    # ---------------- Google Cloud ----------------
    Source("gcp-gcve-2025", "https://cloud.google.com/blog/products/infrastructure/details-of-google-cloud-gcve-incident",
           "Google Cloud", "2025-06-01", "GCVE", "config",
           "Details of the Google Cloud GCVE incident"),

    # ---------------- Microsoft Azure ----------------
    Source("azure-storage-2014", "https://azure.microsoft.com/en-us/blog/update-on-azure-storage-service-interruption/",
           "Microsoft Azure", "2014-11-19", "Azure Storage", "config",
           "Update on Azure Storage service interruption"),

    # ---------------- Other engineering teams ----------------
    Source("sentry-2015-07-20", "https://blog.sentry.io/transaction-id-wraparound-in-postgres/",
           "Sentry", "2015-07-20", "PostgreSQL", "database",
           "Transaction ID wraparound in Postgres"),
    Source("honeycomb-2023-07-25", "https://www.honeycomb.io/blog/incident-review-what-comes-up-must-first-go-down/",
           "Honeycomb", "2023-07-25", "ingest pipeline", "software",
           "Incident review: what comes up must first go down"),
    Source("incidentio-2022-11-18", "https://incident.io/blog/intermittent-downtime",
           "incident.io", "2022-11-18", "async workers", "software",
           "Intermittent downtime after a poison pill event"),
    Source("datadog-2020-09-24", "https://www.datadoghq.com/blog/2020-09-25-infrastructure-connectivity-issue/",
           "Datadog", "2020-09-24", "service discovery", "config",
           "Infrastructure connectivity issue"),
    Source("gocardless-2017-10-10", "https://gocardless.com/blog/incident-review-api-and-dashboard-outage-on-10th-october/",
           "GoCardless", "2017-10-10", "database cluster", "config",
           "Incident review: API and Dashboard outage on 10 October 2017"),
    Source("slack-2021-01-04", "https://slack.engineering/slacks-outage-on-january-4th-2021/",
           "Slack", "2021-01-04", "traffic gateway", "network",
           "Slack's outage on January 4th 2021"),
    Source("slack-2022-02-22", "https://slack.engineering/slacks-incident-on-2-22-22/",
           "Slack", "2022-02-22", "cache cluster", "capacity",
           "Slack's incident on 22 February 2022"),
]


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def html_to_text(html: str, selectors: list) -> tuple:
    """
    Return (text, selector_used).

    Try each selector in order. The first selector that gives text above the
    word floor wins. If no selector reaches the floor, return the longest text
    that was found, so the caller can report the true size.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()

    best_text = ""
    best_selector = "none"

    ordered = list(selectors) + [s for s in DEFAULT_SELECTORS if s not in selectors]
    for selector in ordered:
        node = soup.select_one(selector)
        if node is None:
            continue
        text = node.get_text("\n", strip=True)
        text = clean_text(text)
        if len(text.split()) > len(best_text.split()):
            best_text, best_selector = text, selector
        if len(text.split()) >= WORD_FLOOR:
            return text, selector

    return best_text, best_selector


def clean_text(text: str) -> str:
    """Remove the noise that survives the tag removal."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Social and navigation noise.
        if re.fullmatch(r"[\W_]+", line):
            continue
        if len(line) <= 2:
            continue
        lines.append(line)
    out = "\n\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def is_junk(text: str) -> str:
    """Return a reason when the text is clearly not an article, else ''."""
    low = text.lower()
    if len(text.split()) < WORD_FLOOR:
        return f"only {len(text.split())} words"
    markers = [
        "click here to be redirected",
        "enable javascript",
        "please enable js",
        "check back here to view the current status",
    ]
    for marker in markers:
        if marker in low:
            return f"boilerplate marker: {marker!r}"
    return ""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def fetch(url: str) -> str:
    """Download one URL. Retry on a transient failure."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code in (403, 404, 410):
                break                      # a retry will not help
        except Exception as exc:           # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < RETRIES:
            time.sleep(1.5 * attempt)
    raise RuntimeError(last_error or "unknown error")


def build_markdown(source: Source, body: str, word_count: int) -> str:
    """Make the document. The header holds the metadata, so retrieval can cite it."""
    return (
        f"# {source.title}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Organization | {source.org} |\n"
        f"| Date | {source.published} |\n"
        f"| System | {source.system} |\n"
        f"| Failure category | {source.failure} |\n"
        f"| Source | {source.url} |\n"
        f"| Words | {word_count} |\n\n"
        f"---\n\n"
        f"{body}\n"
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the IncidentLens corpus.")
    parser.add_argument("--force", action="store_true", help="download again and overwrite")
    parser.add_argument("--only", default="", help="only sources whose id contains this text")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SOURCES if args.only in s.sid] if args.only else SOURCES

    print("=" * 78)
    print(" IncidentLens corpus build")
    print(f" sources selected : {len(selected)}")
    print(f" output directory : {OUT_DIR}")
    print(f" word floor       : {WORD_FLOOR}")
    print("=" * 78)

    results = []
    started = time.time()

    for index, source in enumerate(selected, start=1):
        dest = OUT_DIR / f"{source.sid}.md"
        record = {
            "sid": source.sid, "url": source.url, "org": source.org,
            "published": source.published, "system": source.system,
            "failure": source.failure, "title": source.title,
            "file": dest.name,
        }

        if dest.exists() and not args.force:
            words = len(dest.read_text(encoding="utf-8").split())
            record.update(status="cached", words=words)
            results.append(record)
            print(f"[{index:>2}/{len(selected)}] CACHED  {words:>6} words  {source.sid}")
            continue

        try:
            html = fetch(source.url)
            body, selector = html_to_text(html, source.selectors)
            reason = is_junk(body)
            if reason:
                record.update(status="rejected", words=len(body.split()), note=reason)
                print(f"[{index:>2}/{len(selected)}] REJECT  {reason:<34} {source.sid}")
                results.append(record)
                continue

            words = len(body.split())
            dest.write_text(build_markdown(source, body, words), encoding="utf-8")
            record.update(status="ok", words=words, selector=selector,
                          bytes=dest.stat().st_size)
            print(f"[{index:>2}/{len(selected)}] OK      {words:>6} words  {selector:<16} {source.sid}")
        except Exception as exc:  # noqa: BLE001
            record.update(status="failed", note=str(exc))
            print(f"[{index:>2}/{len(selected)}] FAIL    {exc:<34} {source.sid}")

        results.append(record)
        time.sleep(DELAY_BETWEEN)

    # ---------------- report ----------------
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    bad = [r for r in results if r["status"] not in ("ok", "cached")]
    total_words = sum(r.get("words", 0) for r in ok)

    print("\n" + "=" * 78)
    print(f" documents : {len(ok)}")
    print(f" failures  : {len(bad)}")
    print(f" words     : {total_words:,}")
    print(f" elapsed   : {time.time() - started:.1f} s")
    print("=" * 78)

    META_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# IncidentLens corpus build report", "",
        f"Sources selected: **{len(selected)}**  ",
        f"Documents built: **{len(ok)}**  ",
        f"Failures: **{len(bad)}**  ",
        f"Total words: **{total_words:,}**  ",
        f"Build date: {date.today().isoformat()}", "",
        "## Documents", "",
        "| id | organization | date | system | failure | words |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(ok, key=lambda x: x["sid"]):
        lines.append(
            f"| `{r['sid']}` | {r['org']} | {r['published']} | {r['system']} | {r['failure']} | {r.get('words', 0):,} |"
        )
    if bad:
        lines += ["", "## Failures", "", "| id | url | reason |", "|---|---|---|"]
        for r in bad:
            lines.append(f"| `{r['sid']}` | {r['url']} | {r.get('note', '')} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f" metadata -> {META_PATH}")
    print(f" report   -> {REPORT_PATH}")

    if bad:
        print("\nThe sources below did not build. Replace them in the manifest or accept a smaller corpus:")
        for r in bad:
            print(f"  - {r['sid']}: {r.get('note', '')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
