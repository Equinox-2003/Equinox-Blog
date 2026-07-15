from __future__ import annotations

import argparse
import datetime as dt
import html
import pathlib
import re
import socket
import time
import urllib.error
import urllib.request

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
MAX_PAGES_PER_STATUS = 100
STATUS_PAGES = [
    ("do", "在看"),
    ("collect", "看过"),
    ("wish", "想看"),
    ("on_hold", "搁置"),
    ("dropped", "抛弃"),
]
ITEM_RE = re.compile(r'<li id="item_(?P<subject_id>\d+)" class="item [^"]*?clearit" ?>(?P<body>.*?)</li>', re.S)
TITLE_RE = re.compile(r'<a href="(?P<url>/subject/\d+)" class="l">(?P<title>.*?)</a>', re.S)
ORIGINAL_TITLE_RE = re.compile(r'<small class="grey">(?P<original_title>.*?)</small>', re.S)
COVER_RE = re.compile(r'<img src="(?P<cover>[^"]+)" class="cover"', re.S)
DATE_RE = re.compile(r'<span class="tip_j">(?P<date>[^<]+)</span>', re.S)
RATING_RE = re.compile(r'stars(?P<rating>\d+)', re.S)


class BangumiFetchError(RuntimeError):
    pass


def build_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def fetch_html(
    url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )
    attempts = retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = retry_delay * 2 ** (attempt - 1)
            print(
                f"Request failed ({attempt}/{attempts}) for {url}: {exc}; "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)

    proxy_hint = " Configure --proxy or HTTPS_PROXY if this network requires a proxy."
    raise BangumiFetchError(
        f"Failed to fetch {url} after {attempts} attempts: {last_error}.{proxy_hint}"
    )


def clean_text(value: str) -> str:
    text = re.sub(r"<.*?>", "", value)
    text = html.unescape(text)
    return " ".join(text.split())


def normalize_url(url: str) -> str:
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://bangumi.tv{url}"
    return url


def parse_items(page_html: str, status: str, status_label: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for match in ITEM_RE.finditer(page_html):
        body = match.group("body")
        title_match = TITLE_RE.search(body)
        cover_match = COVER_RE.search(body)
        if not title_match or not cover_match:
            continue

        item: dict[str, object] = {
            "subject_id": match.group("subject_id"),
            "title": clean_text(title_match.group("title")),
            "url": normalize_url(title_match.group("url")),
            "cover": normalize_url(cover_match.group("cover")),
            "status": status,
            "status_label": status_label,
        }

        original_title_match = ORIGINAL_TITLE_RE.search(body)
        if original_title_match:
            original_title = clean_text(original_title_match.group("original_title"))
            if original_title:
                item["original_title"] = original_title

        date_match = DATE_RE.search(body)
        if date_match:
            item["date"] = clean_text(date_match.group("date"))

        rating_match = RATING_RE.search(body)
        if rating_match:
            item["rating"] = int(rating_match.group("rating"))

        items.append(item)
    return items


def yaml_quote(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def dump_yaml(user_id: str, items: list[dict[str, object]]) -> str:
    today = dt.date.today().isoformat()
    lines = [
        "user:",
        f"  id: {yaml_quote(user_id)}",
        f"  profile: {yaml_quote(f'https://bangumi.tv/user/{user_id}')}",
        f"  collection_url: {yaml_quote(f'https://bangumi.tv/anime/list/{user_id}')}",
        f"  updated_at: {yaml_quote(today)}",
        "items:",
    ]

    for item in items:
        lines.append(f"  - title: {yaml_quote(item['title'])}")
        if item.get("original_title"):
            lines.append(f"    original_title: {yaml_quote(item['original_title'])}")
        lines.append(f"    url: {yaml_quote(item['url'])}")
        lines.append(f"    cover: {yaml_quote(item['cover'])}")
        lines.append(f"    status: {yaml_quote(item['status'])}")
        lines.append(f"    status_label: {yaml_quote(item['status_label'])}")
        if item.get("date"):
            lines.append(f"    date: {yaml_quote(item['date'])}")
        if item.get("rating") is not None:
            lines.append(f"    rating: {item['rating']}")
    return "\n".join(lines) + "\n"


def fetch_status_items(
    user_id: str,
    status: str,
    status_label: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> list[dict[str, object]]:
    items_by_id: dict[str, dict[str, object]] = {}
    page = 1
    while page <= MAX_PAGES_PER_STATUS:
        url = f"https://bangumi.tv/anime/list/{user_id}/{status}"
        if page > 1:
            url = f"{url}?page={page}"
        page_html = fetch_html(url, opener, timeout, retries, retry_delay)
        page_items = parse_items(page_html, status, status_label)
        if not page_items:
            break
        previous_count = len(items_by_id)
        for item in page_items:
            items_by_id[str(item["subject_id"])] = item
        if len(items_by_id) == previous_count:
            break
        page += 1
    else:
        raise BangumiFetchError(
            f"Stopped after {MAX_PAGES_PER_STATUS} pages while fetching status {status}."
        )
    return list(items_by_id.values())


def refresh_bangumi(
    user_id: str,
    output_path: pathlib.Path,
    proxy: str | None,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> int:
    opener = build_opener(proxy)
    all_items: list[dict[str, object]] = []
    for status, status_label in STATUS_PAGES:
        all_items.extend(
            fetch_status_items(
                user_id,
                status,
                status_label,
                opener,
                timeout,
                retries,
                retry_delay,
            )
        )

    if not all_items:
        raise BangumiFetchError(f"No Bangumi items fetched for user {user_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(dump_yaml(user_id, all_items), encoding="utf-8")
    temporary_path.replace(output_path)
    return len(all_items)


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Refresh Bangumi anime collection data for Hugo.")
    parser.add_argument("--user-id", default="1214444", help="Bangumi user id")
    parser.add_argument("--output", default=str(root / "data" / "bangumi.yml"), help="Output YAML path")
    parser.add_argument(
        "--proxy",
        help="HTTP(S) proxy, for example http://127.0.0.1:7890; defaults to HTTPS_PROXY when set",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries after the first request")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="Initial retry delay in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0 or args.retries < 0 or args.retry_delay < 0:
        raise SystemExit("--timeout must be positive; --retries and --retry-delay cannot be negative.")
    output_path = pathlib.Path(args.output)
    try:
        count = refresh_bangumi(
            args.user_id,
            output_path,
            args.proxy,
            args.timeout,
            args.retries,
            args.retry_delay,
        )
    except BangumiFetchError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {count} Bangumi items to {output_path}")


if __name__ == "__main__":
    main()
