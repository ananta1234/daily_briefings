import os
import re
import json
import calendar
import feedparser
import yaml
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import requests
import anthropic
from jinja2 import Environment, FileSystemLoader

ET = ZoneInfo("America/New_York")
FETCH_TIMEOUT = 15  # seconds per feed
MAX_ARTICLES_FOR_AI = 100


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def clean_html(text):
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(text.split())


def fmt_date(dt):
    """Format a datetime without leading zeros (cross-platform safe)."""
    if not dt:
        return ""
    dt_et = dt.astimezone(ET)
    month = str(dt_et.month)
    day = str(dt_et.day)
    hour = str(int(dt_et.strftime("%I")))
    minute = dt_et.strftime("%M")
    ampm = dt_et.strftime("%p")
    return f"{month}/{day} {hour}:{minute} {ampm} ET"


def parse_entry_date(entry):
    """Extract a UTC-aware datetime from a feedparser entry."""
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime.fromtimestamp(calendar.timegm(val), tz=timezone.utc)
            except Exception:
                pass
    return None


def fetch_feed(source):
    """Fetch one RSS feed. Returns (name, articles, error_or_None)."""
    try:
        resp = requests.get(
            source["url"],
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "DailyBriefing/1.0"},
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        articles = []

        for entry in feed.entries:
            published = parse_entry_date(entry)
            if published and published < cutoff:
                continue

            description = clean_html(entry.get("summary", "") or "")[:400]
            title = clean_html(entry.get("title", "") or "Untitled")
            url = entry.get("link", "")

            if not url:
                continue

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "description": description,
                    "published": published.isoformat() if published else None,
                    "published_display": fmt_date(published),
                    "source": source["name"],
                }
            )

        return source["name"], articles, None

    except Exception as e:
        return source["name"], [], str(e)


def fetch_all_feeds(sources):
    articles, errors = [], []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_feed, s): s for s in sources}
        for future in as_completed(futures):
            name, feed_articles, error = future.result()
            if error:
                errors.append(f"{name}: {error}")
                print(f"[WARN] {name}: {error}")
            else:
                print(f"[OK]   {name}: {len(feed_articles)} articles")
                articles.extend(feed_articles)
    return articles, errors


def deduplicate(articles):
    seen = set()
    result = []
    for a in articles:
        key = a["url"] or a["title"]
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result


def get_ai_analysis(articles, interests):
    """
    Send articles to Claude and get back:
    - highlights: top 5 articles with why_it_matters explanations
    - scores: dict of url -> relevance score (1-10)

    Returns (highlights, scores). On any failure, returns ([], {}).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not articles:
        print("[WARN] Skipping AI analysis (no API key or no articles)")
        return [], {}

    client = anthropic.Anthropic(api_key=api_key)
    articles_for_ai = articles[:MAX_ARTICLES_FOR_AI]

    interests_str = "\n".join(f"- {i}" for i in interests)
    articles_list = "\n\n".join(
        f"[{i}] {a['title']} ({a['source']})\n{a['description'][:200]}"
        for i, a in enumerate(articles_for_ai)
    )

    prompt = f"""You are a personal news curator. The reader's interests:
{interests_str}

Today's articles (indexed 0 to {len(articles_for_ai) - 1}):
{articles_list}

Return a JSON object (no other text) with:
{{
  "highlights": [
    {{
      "index": <int>,
      "why_it_matters": "<1-2 sentences: why this is relevant to the reader's specific interests>"
    }}
  ],
  "scores": {{
    "<index as string>": <relevance score 1-10>
  }}
}}

Pick the 5 most relevant articles for highlights. Score all {len(articles_for_ai)} articles."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        highlights = []
        for h in result.get("highlights", []):
            idx = h.get("index")
            if idx is not None and 0 <= idx < len(articles_for_ai):
                article = articles_for_ai[idx].copy()
                article["why_it_matters"] = h.get("why_it_matters", "")
                highlights.append(article)

        scores = {}
        for idx_str, score in result.get("scores", {}).items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(articles_for_ai):
                    scores[articles_for_ai[idx]["url"]] = int(score)
            except (ValueError, TypeError):
                pass

        return highlights, scores

    except Exception as e:
        print(f"[WARN] AI analysis failed: {e}. Showing articles without scoring.")
        return [], {}


def render(articles, highlights, scores, errors, interests, generated_at, by_source):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(script_dir), autoescape=True)
    template = env.get_template("template.html")

    highlight_urls = {a["url"] for a in highlights}

    return template.render(
        highlights=highlights,
        by_source=by_source,
        scores=scores,
        highlight_urls=highlight_urls,
        errors=errors,
        interests=interests,
        generated_at=generated_at,
        total_articles=len(articles),
    )


def main():
    config = load_config()
    interests = config["interests"]
    sources = config["sources"]

    print(f"[INFO] Fetching {len(sources)} feeds...")
    articles, errors = fetch_all_feeds(sources)
    articles = deduplicate(articles)
    articles.sort(key=lambda a: a["published"] or "", reverse=True)
    print(f"[INFO] {len(articles)} unique articles after dedup")

    print("[INFO] Running AI analysis...")
    highlights, scores = get_ai_analysis(articles, interests)
    print(f"[INFO] {len(highlights)} highlights, {len(scores)} scored")

    # Group by source, sort each group by relevance then date
    by_source = {}
    for a in articles:
        by_source.setdefault(a["source"], []).append(a)
    for src in by_source:
        by_source[src].sort(
            key=lambda a: (scores.get(a["url"], 0), a["published"] or ""),
            reverse=True,
        )

    now_et = datetime.now(ET)
    # %-d / %-I strips leading zeros on Linux (GitHub Actions); use int() for safety
    day = str(now_et.day)
    hour = str(int(now_et.strftime("%I")))
    generated_at = now_et.strftime(f"%A, %B {day}, %Y at {hour}:%M %p ET")

    html = render(articles, highlights, scores, errors, interests, generated_at, by_source)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Done — {out_path}")


if __name__ == "__main__":
    main()
