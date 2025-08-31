import json, os
from pathlib import Path
from news_fetcher import fetch_trending_content, normalize_headline, load_processed, save_processed, fuzzy_already_processed
from script_generator import build_news_script, generate_caption_and_hashtags
from media_utils import tts_to_file, build_timed_srt, get_audio_duration

CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("videos")
CONTENT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def stage_fetch_news(limit=2):
    items = fetch_trending_content(limit)
    processed = load_processed()
    new_items=[]
    for doc in items:
        raw = doc["title"]
        norm = normalize_headline(raw)
        if fuzzy_already_processed(norm, processed): continue
        processed.append(norm)
        new_items.append(doc)
    save_processed(processed)
    (CONTENT_DIR / "fetched_items.json").write_text(json.dumps(new_items,indent=2))
    print(f"Stage1: fetched {len(new_items)} items")
    return new_items

def stage_build_scripts(news_items):
    for doc in news_items:
        headline = doc["title"]
        doc["script"] = build_news_script(headline)
        doc["caption"] = generate_caption_and_hashtags(headline, doc["script"])
    (CONTENT_DIR / "fetched_items.json").write_text(json.dumps(news_items,indent=2))
    print("Stage2: built scripts & captions")
    return news_items

def stage_create_media(news_items):
    for i, doc in enumerate(news_items, start=1):
        headline, script = doc["title"], doc.get("script", doc["title"])
        voice_path = CONTENT_DIR / f"voice_{i}.mp3"
        tts_to_file(script, str(voice_path))
        duration = get_audio_duration(str(voice_path)) or max(10,len(script.split())/2)
        srt_path = CONTENT_DIR / f"subs_{i}.srt"
        build_timed_srt(script, str(srt_path), duration)
    print("Stage3: generated audio & SRT")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=int,default=1)
    parser.add_argument("--limit",type=int,default=2)
    args=parser.parse_args()
    stage=args.stage
    news_items=None

    if stage==1:
        stage_fetch_news(limit=args.limit)
    elif stage==2:
        news_items=json.loads((CONTENT_DIR / "fetched_items.json").read_text())
        stage_build_scripts(news_items)
    elif stage==3:
        news_items=json.loads((CONTENT_DIR / "fetched_items.json").read_text())
        stage_create_media(news_items)
