import os
import json
import re
import requests
from pathlib import Path
from difflib import SequenceMatcher

CONTENT_DIR = Path("content")
STATE_FILE = CONTENT_DIR / "processed.json"
MAX_PROCESSED = 5000
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FUZZY_THRESHOLD = 0.82

CONTENT_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]

def normalize_headline(title: str) -> str:
    t = title or ""
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")
    t = re.sub(r"\s+[-|]\s*[^-|\n]{1,80}$", "", t)
    t = re.sub(r'^(the|a|an)\s+[\w\s]{1,40}\.?[:\-]\s*', '', t, flags=re.I)
    t = re.sub(r'(\b[A-Za-z]{2,}\b)(\s+\1)+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t.lower()

def load_processed():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except:
            return []
    return []

def save_processed(items):
    items = items[-MAX_PROCESSED:]
    STATE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def fuzzy_already_processed(norm_title, processed):
    if norm_title in processed:
        return True
    for p in processed:
        if SequenceMatcher(None, norm_title, p).ratio() >= FUZZY_THRESHOLD:
            return True
    return False

def fetch_trending_content(limit=5):
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not set")
    url = "https://newsapi.org/v2/top-headlines"
    resp = requests.get(url, params={"language":"en","pageSize":limit,"apiKey":NEWS_API_KEY}, timeout=15)
    resp.raise_for_status()
    out, seen = [], set()
    for a in resp.json().get("articles", []):
        t = (a.get("title") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append({"title": t, "source": a.get("source", {}).get("name", ""), "url": a.get("url","")})
    return out[:limit]
