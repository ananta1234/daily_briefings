import os
import re
import json
import random
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
FETCH_TIMEOUT = 15
LOOKBACK_DAYS = 30
MAX_ARTICLES_FOR_AI = 100   # total sent to Claude per run
RECENT_ARTICLES = 50        # most recent N always included
OLDER_SAMPLE = 50           # random sample from older articles
MAX_SEEN_HIGHLIGHTS = 500   # cap seen history so state.json stays small
GITHUB_REPO = "ananta1234/daily_briefings"
GITHUB_BRANCH = "main"
GITHUB_CONFIG_URL = f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/config.yaml"


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def state_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "state.json")


def load_state():
    path = state_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
                # ensure expected keys exist
                data.setdefault("seen_highlights", [])
                data.setdefault("pins", [])
                return data
        except Exception as e:
            print(f"[WARN] Could not load state.json: {e}")
    return {"seen_highlights": [], "pins": []}


def save_state(state):
    path = state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(text.split())


def fmt_date(dt):
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
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime.fromtimestamp(calendar.timegm(val), tz=timezone.utc)
            except Exception:
                pass
    return None


def fetch_feed(source):
    try:
        resp = requests.get(
            source["url"],
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "DailyBriefing/1.0"},
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
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

            articles.append({
                "title": title,
                "url": url,
                "description": description,
                "published": published.isoformat() if published else None,
                "published_display": fmt_date(published),
                "source": source["name"],
                "topic": "General Tech",
            })

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


def select_for_ai(articles):
    """
    Take 50 most recent + random sample of 50 older ones.
    This ensures both recency and discovery of older gems.
    """
    sorted_articles = sorted(articles, key=lambda a: a["published"] or "", reverse=True)
    recent = sorted_articles[:RECENT_ARTICLES]
    older = sorted_articles[RECENT_ARTICLES:]
    sampled_older = random.sample(older, min(OLDER_SAMPLE, len(older))) if older else []
    combined = recent + sampled_older
    # Sort by date so Claude sees temporal context
    combined.sort(key=lambda a: a["published"] or "", reverse=True)
    return combined[:MAX_ARTICLES_FOR_AI]


def get_ai_analysis(articles, interests, topics, seen_highlights):
    """
    Returns (highlights, scores, topic_map).
    On failure returns ([], {}, {}) — site renders without AI features.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not articles:
        print("[WARN] Skipping AI analysis (no API key or no articles)")
        return [], {}, {}

    client = anthropic.Anthropic(api_key=api_key)
    articles_for_ai = select_for_ai(articles)

    interests_str = "\n".join(f"- {i}" for i in interests)
    topics_str = ", ".join(f'"{t}"' for t in topics)
    articles_list = "\n\n".join(
        f"[{i}] {a['title']} ({a['source']}, {a['published_display'] or 'undated'})\n{a['description'][:200]}"
        for i, a in enumerate(articles_for_ai)
    )

    seen_str = ""
    if seen_highlights:
        seen_str = f"\nAlready surfaced to the reader (avoid repeating in highlights unless truly exceptional):\n" + \
                   "\n".join(f"- {url}" for url in seen_highlights[-100:])

    prompt = f"""You are a personal news curator for a specific reader. Their interests:
{interests_str}

Available topic categories: {topics_str}
{seen_str}

Articles available (indexed 0 to {len(articles_for_ai) - 1}):
{articles_list}

Return a JSON object (no other text) with exactly these keys:
{{
  "highlights": [
    {{
      "index": <int>,
      "why_it_matters": "<1-2 sentences: why this is significant and relevant to the reader's specific interests>"
    }}
  ],
  "scores": {{
    "<index as string>": <relevance score 1-10>
  }},
  "topics": {{
    "<index as string>": "<one of the available topic categories>"
  }}
}}

Rules:
- highlights: pick the 5 most significant AND relevant articles the reader hasn't seen yet. Articles can be from any date — prioritize quality and relevance. Avoid repeating previously surfaced URLs unless the article is truly exceptional.
- scores: score all {len(articles_for_ai)} articles for relevance (1-10)
- topics: assign every article to exactly one topic category"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
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

        topic_map = {}
        for idx_str, topic in result.get("topics", {}).items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(articles_for_ai):
                    topic_map[articles_for_ai[idx]["url"]] = topic
            except (ValueError, TypeError):
                pass

        return highlights, scores, topic_map

    except Exception as e:
        print(f"[WARN] AI analysis failed: {e}. Rendering without AI features.")
        return [], {}, {}


def main():
    config = load_config()
    interests = config["interests"]
    topics = config.get("topics", ["General Tech"])
    sources = config["sources"]

    state = load_state()
    seen_highlights = state.get("seen_highlights", [])
    print(f"[INFO] {len(seen_highlights)} previously seen highlights loaded")

    print(f"[INFO] Fetching {len(sources)} feeds (last {LOOKBACK_DAYS} days)...")
    articles, errors = fetch_all_feeds(sources)
    articles = deduplicate(articles)
    articles.sort(key=lambda a: a["published"] or "", reverse=True)
    print(f"[INFO] {len(articles)} unique articles after dedup")

    print("[INFO] Running AI analysis...")
    highlights, scores, topic_map = get_ai_analysis(articles, interests, topics, seen_highlights)
    print(f"[INFO] {len(highlights)} highlights, {len(scores)} scored, {len(topic_map)} tagged")

    # Update seen highlights — add new ones, cap list size
    new_seen = seen_highlights + [a["url"] for a in highlights]
    state["seen_highlights"] = new_seen[-MAX_SEEN_HIGHLIGHTS:]
    save_state(state)
    print(f"[INFO] state.json updated ({len(state['seen_highlights'])} seen highlights)")

    # Apply topic tags
    for a in articles:
        a["topic"] = topic_map.get(a["url"], "General Tech")

    # Group by source, sort by relevance then recency
    by_source = {}
    for a in articles:
        by_source.setdefault(a["source"], []).append(a)
    for src in by_source:
        by_source[src].sort(
            key=lambda a: (scores.get(a["url"], 0), a["published"] or ""),
            reverse=True,
        )

    now_et = datetime.now(ET)
    day = str(now_et.day)
    hour = str(int(now_et.strftime("%I")))
    generated_at = now_et.strftime(f"%A, %B {day}, %Y at {hour}:%M %p ET")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(script_dir), autoescape=True)
    template = env.get_template("template.html")
    highlight_urls = {a["url"] for a in highlights}

    html = template.render(
        highlights=highlights,
        by_source=by_source,
        scores=scores,
        highlight_urls=highlight_urls,
        errors=errors,
        interests=interests,
        topics=topics,
        sources=sources,
        generated_at=generated_at,
        total_articles=len(articles),
        github_config_url=GITHUB_CONFIG_URL,
        github_repo=GITHUB_REPO,
        github_branch=GITHUB_BRANCH,
    )

    out_dir = os.path.join(script_dir, "docs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Done.")


if __name__ == "__main__":
    main()
