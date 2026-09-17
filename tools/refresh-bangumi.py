from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = (
    "EquinoxBlog/1.0 (+https://blog.equinox.wiki) "
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
)
DEFAULT_API_BASE_URL = "https://api.bgm.tv/v0"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
API_PAGE_SIZE = 50
MAX_PAGES_PER_STATUS = 100

SUBJECT_TYPES = (
    {"key": "anime", "label": "动画", "subject_type": 2, "path": "anime"},
    {"key": "game", "label": "游戏", "subject_type": 4, "path": "game"},
    {"key": "book", "label": "书籍", "subject_type": 1, "path": "book"},
    {"key": "real", "label": "电视剧", "subject_type": 6, "path": "real"},
)

STATUS_LABELS = {
    "anime": {"do": "在看", "collect": "看过", "wish": "想看", "on_hold": "搁置", "dropped": "抛弃"},
    "game": {"do": "在玩", "collect": "玩过", "wish": "想玩", "on_hold": "搁置", "dropped": "抛弃"},
    "book": {"do": "在读", "collect": "读过", "wish": "想读", "on_hold": "搁置", "dropped": "抛弃"},
    "real": {"do": "在看", "collect": "看过", "wish": "想看", "on_hold": "搁置", "dropped": "抛弃"},
}
STATUS_PAGES = ("do", "collect", "wish", "on_hold", "dropped")

ITEM_RE = re.compile(
    r'<li\s+id="item_(?P<subject_id>\d+)"[^>]*class="[^">]*\bitem\b[^">]*"[^>]*>'
    r"(?P<body>.*?)</li>",
    re.S,
)
TITLE_RE = re.compile(
    r'<a\s+href="(?P<url>/subject/\d+)"[^>]*class="[^"]*\bl\b[^"]*"[^>]*>'
    r"(?P<title>.*?)</a>",
    re.S,
)
ORIGINAL_TITLE_RE = re.compile(r'<small class="grey">(?P<original_title>.*?)</small>', re.S)
DATE_RE = re.compile(r'<span class="tip_j">(?P<date>[^<]+)</span>', re.S)
RATING_RE = re.compile(r"stars(?P<rating>\d+)", re.S)
COVER_RE = (
    re.compile(r'<img[^>]*src="(?P<cover>[^"]+)"[^>]*class="[^"]*\bcover\b[^"]*"', re.S),
    re.compile(r'<img[^>]*class="[^"]*\bcover\b[^"]*"[^>]*src="(?P<cover>[^"]+)"', re.S),
)


class BangumiFetchError(RuntimeError):
    pass


def build_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def fetch_bytes(
    url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
    accept: str,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": USER_AGENT,
        },
    )
    attempts = retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.read()
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

    raise BangumiFetchError(
        f"Failed to fetch {url} after {attempts} attempts: {last_error}."
    )


def fetch_json(
    url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> dict[str, object]:
    payload = fetch_bytes(
        url, opener, timeout, retries, retry_delay, "application/json"
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BangumiFetchError(f"Invalid JSON returned by {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise BangumiFetchError(f"Unexpected JSON response from {url}: expected an object.")
    return value


def fetch_html(
    url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> str:
    payload = fetch_bytes(
        url,
        opener,
        timeout,
        retries,
        retry_delay,
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    )
    return payload.decode("utf-8", errors="ignore")


def clean_text(value: object) -> str:
    text = re.sub(r"<.*?>", "", str(value))
    text = html.unescape(text)
    return " ".join(text.split())


def normalize_url(url: str) -> str:
    value = str(url)
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("/"):
        return f"https://bangumi.tv{value}"
    return value


def first_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = clean_text(value)
        if text:
            return text
    return ""


def get_cover(subject: dict[str, object], entry: dict[str, object]) -> str:
    images = subject.get("images") or entry.get("images")
    if isinstance(images, dict):
        for key in ("large", "common", "medium", "grid", "small"):
            if images.get(key):
                return normalize_url(str(images[key]))
    if isinstance(images, str):
        return normalize_url(images)
    return ""


def get_rating(entry: dict[str, object], subject: dict[str, object]) -> int | None:
    rating = entry.get("rating") or subject.get("rating")
    if isinstance(rating, dict):
        rating = rating.get("score")
    if rating is None:
        rating = entry.get("rate")
    try:
        score = int(float(rating))
    except (TypeError, ValueError):
        return None
    return score if score > 0 else None


def get_tags(subject: dict[str, object]) -> list[str]:
    tags = subject.get("tags")
    if not isinstance(tags, list):
        return []
    result: list[str] = []
    for tag in tags:
        value = tag.get("name") if isinstance(tag, dict) else tag
        text = clean_text(value) if value is not None else ""
        if text and text not in result:
            result.append(text)
    return result[:12]


def parse_api_items(
    response: dict[str, object], category_key: str, status: str
) -> list[dict[str, object]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise BangumiFetchError("Bangumi API response does not contain a data list.")

    items: list[dict[str, object]] = []
    for entry_value in data:
        if not isinstance(entry_value, dict):
            continue
        entry = entry_value
        subject_value = entry.get("subject")
        subject = subject_value if isinstance(subject_value, dict) else {}
        subject_id = entry.get("subject_id") or subject.get("id")
        if subject_id is None:
            continue

        title = first_text(
            subject.get("name_cn"),
            subject.get("name"),
            entry.get("name_cn"),
            entry.get("name"),
        )
        if not title:
            continue

        item: dict[str, object] = {
            "subject_id": str(subject_id),
            "title": title,
            "url": normalize_url(str(subject.get("url") or f"/subject/{subject_id}")),
            "cover": get_cover(subject, entry),
            "status": status,
            "status_label": STATUS_LABELS[category_key][status],
        }

        original_title = first_text(subject.get("name"), entry.get("name"))
        if original_title and original_title != title:
            item["original_title"] = original_title

        date = first_text(entry.get("updated_at"), entry.get("created_at"), entry.get("date"))
        if date:
            item["date"] = date[:10]

        summary = first_text(subject.get("short_summary"), subject.get("summary"))
        if summary:
            item["summary"] = summary

        tags = get_tags(subject)
        if tags:
            item["tags"] = tags

        rating = get_rating(entry, subject)
        if rating is not None:
            item["rating"] = rating
        items.append(item)
    return items


def fetch_api_status_items(
    user_id: str,
    category: dict[str, object],
    status: str,
    api_base_url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    offset = 0
    for _ in range(MAX_PAGES_PER_STATUS):
        query = urllib.parse.urlencode(
            {
                "subject_type": category["subject_type"],
                "type": status,
                "limit": API_PAGE_SIZE,
                "offset": offset,
            }
        )
        url = (
            f"{api_base_url.rstrip('/')}/users/{urllib.parse.quote(user_id, safe='')}"
            f"/collections?{query}"
        )
        response = fetch_json(url, opener, timeout, retries, retry_delay)
        raw_data = response.get("data")
        if not isinstance(raw_data, list) or not raw_data:
            break
        items.extend(parse_api_items(response, str(category["key"]), status))
        if len(raw_data) < API_PAGE_SIZE:
            break
        offset += API_PAGE_SIZE
    else:
        raise BangumiFetchError(
            f"Stopped after {MAX_PAGES_PER_STATUS} pages while fetching "
            f"{category['key']} status {status}."
        )
    return items


def parse_html_items(page_html: str, category_key: str, status: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item_match in ITEM_RE.finditer(page_html):
        body = item_match.group("body")
        title_match = TITLE_RE.search(body)
        cover_match = next(
            (match for pattern in COVER_RE if (match := pattern.search(body))), None
        )
        if not title_match:
            continue

        item: dict[str, object] = {
            "subject_id": item_match.group("subject_id"),
            "title": clean_text(title_match.group("title")),
            "url": normalize_url(title_match.group("url")),
            "cover": normalize_url(cover_match.group("cover")) if cover_match else "",
            "status": status,
            "status_label": STATUS_LABELS[category_key][status],
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


def fetch_html_status_items(
    user_id: str,
    category: dict[str, object],
    status: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for page in range(1, MAX_PAGES_PER_STATUS + 1):
        url = f"https://bangumi.tv/{category['path']}/list/{user_id}/{status}"
        if page > 1:
            url = f"{url}?page={page}"
        page_items = parse_html_items(
            fetch_html(url, opener, timeout, retries, retry_delay),
            str(category["key"]),
            status,
        )
        if not page_items:
            break
        items.extend(page_items)
        if len(page_items) < 24:
            break
    else:
        raise BangumiFetchError(
            f"Stopped after {MAX_PAGES_PER_STATUS} pages while fetching "
            f"{category['key']} status {status}."
        )
    return items


def deduplicate_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for item in items:
        by_id[str(item["subject_id"])] = item
    return list(by_id.values())


def fetch_category(
    user_id: str,
    category: dict[str, object],
    source: str,
    api_base_url: str,
    opener: urllib.request.OpenerDirector,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> list[dict[str, object]]:
    category_key = str(category["key"])
    if source in ("auto", "api"):
        try:
            items: list[dict[str, object]] = []
            for status in STATUS_PAGES:
                items.extend(
                    fetch_api_status_items(
                        user_id,
                        category,
                        status,
                        api_base_url,
                        opener,
                        timeout,
                        retries,
                        retry_delay,
                    )
                )
            return deduplicate_items(items)
        except BangumiFetchError as exc:
            if source == "api":
                raise
            print(f"API source failed for {category_key}: {exc}; trying HTML fallback.")

    items: list[dict[str, object]] = []
    for status in STATUS_PAGES:
        items.extend(
            fetch_html_status_items(
                user_id,
                category,
                status,
                opener,
                timeout,
                retries,
                retry_delay,
            )
        )
    return deduplicate_items(items)


def yaml_quote(value: object) -> str:
    text = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{text}"'


def dump_yaml(
    user_id: str,
    categories: list[tuple[dict[str, object], list[dict[str, object]]]],
) -> str:
    today = dt.date.today().isoformat()
    lines = [
        "user:",
        f"  id: {yaml_quote(user_id)}",
        f"  profile: {yaml_quote(f'https://bangumi.tv/user/{user_id}')}",
        f"  timeline_url: {yaml_quote(f'https://bangumi.tv/user/{user_id}/timeline')}",
        f"  updated_at: {yaml_quote(today)}",
        "categories:",
    ]

    for category, items in categories:
        category_key = str(category["key"])
        collection_url = f"https://bangumi.tv/{category['path']}/list/{user_id}"
        lines.extend(
            [
                f"  {category_key}:",
                f"    key: {yaml_quote(category_key)}",
                f"    label: {yaml_quote(category['label'])}",
                f"    subject_type: {category['subject_type']}",
                f"    collection_url: {yaml_quote(collection_url)}",
            ]
        )
        if not items:
            lines.append("    items: []")
            continue
        lines.append("    items:")
        for item in items:
            lines.append(f"      - subject_id: {yaml_quote(item['subject_id'])}")
            for key in (
                "title",
                "original_title",
                "url",
                "cover",
                "status",
                "status_label",
                "date",
                "summary",
            ):
                if item.get(key):
                    lines.append(f"        {key}: {yaml_quote(item[key])}")
            if item.get("rating") is not None:
                lines.append(f"        rating: {item['rating']}")
            tags = item.get("tags")
            if isinstance(tags, list) and tags:
                lines.append("        tags:")
                lines.extend(f"          - {yaml_quote(tag)}" for tag in tags)
    return "\n".join(lines) + "\n"


def refresh_bangumi(
    user_id: str,
    output_path: pathlib.Path,
    proxy: str | None,
    source: str,
    api_base_url: str,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> int:
    opener = build_opener(proxy)
    fetched_categories: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    total = 0
    for category in SUBJECT_TYPES:
        items = fetch_category(
            user_id,
            category,
            source,
            api_base_url,
            opener,
            timeout,
            retries,
            retry_delay,
        )
        fetched_categories.append((category, items))
        total += len(items)
        print(f"Fetched {len(items)} {category['label']} items.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        dump_yaml(user_id, fetched_categories),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return total


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Refresh Bangumi time-machine collection data for Hugo."
    )
    parser.add_argument("--user-id", default="1214444", help="Bangumi user id")
    parser.add_argument(
        "--output",
        default=str(root / "data" / "bangumi.yml"),
        help="Output YAML path",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "api", "html"),
        default="auto",
        help="Collection source; auto uses API then HTML fallback",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="Bangumi API base URL",
    )
    parser.add_argument(
        "--proxy",
        help="HTTP(S) proxy, for example http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retries after the first request",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Initial retry delay in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0 or args.retries < 0 or args.retry_delay < 0:
        raise SystemExit(
            "--timeout must be positive; --retries and --retry-delay cannot be negative."
        )
    try:
        count = refresh_bangumi(
            args.user_id,
            pathlib.Path(args.output),
            args.proxy,
            args.source,
            args.api_base_url,
            args.timeout,
            args.retries,
            args.retry_delay,
        )
    except BangumiFetchError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Wrote {count} Bangumi items across {len(SUBJECT_TYPES)} categories "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
