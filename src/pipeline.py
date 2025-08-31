import json
from pathlib import Path

# modules
from news_fetcher import fetch_trending_content, normalize_headline, load_processed, save_processed, fuzzy_already_processed
from script_generator import build_news_script, generate_caption_and_hashtags
from media_utils import tts_to_file, build_timed_srt, get_audio_duration, mix_audio, safe_filename

CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("videos")

# ---------------- STAGE 1: fetch & dedupe ----------------
def stage_fetch_news(limit=2):
    items = fetch_trending_content(limit=limit)
    processed = load_processed()
    new_items = []
    for doc in items:
        raw_title = doc["title"]
        norm = normalize_headline(raw_title)
        if fuzzy_already_processed(norm, processed):
            print(f"SKIP (already processed): {raw_title}")
            continue
        processed.append(norm)
        new_items.append(doc)
    save_processed(processed)
    print(f"Fetched {len(new_items)} new items")
    return new_items

# ---------------- STAGE 2: build scripts & captions ----------------
def stage_build_scripts(news_items):
    for doc in news_items:
        headline = doc["title"]
        script = build_news_script(headline)
        caption = generate_caption_and_hashtags(headline, script)
        doc["script"] = script
        doc["caption"] = caption
        print(f"Built script for: {headline}")
    return news_items

# ---------------- STAGE 3: media/audio/video ----------------
def stage_create_media(news_items):
    for i, doc in enumerate(news_items, start=1):
        headline = doc["title"]
        script = doc.get("script", headline)
        # TTS
        voice_path = CONTENT_DIR / f"voice_{i}.mp3"
        tts_to_file(script, str(voice_path))
        duration = get_audio_duration(str(voice_path)) or max(10.0, len(script.split()) / 2.0)
        # SRT
        srt_path = CONTENT_DIR / f"subs_{i}.srt"
        build_timed_srt(script, str(srt_path), total_duration=duration)
        print(f"Generated media for: {headline}")
    return news_items

# ---------------- CLI / pipeline ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="Stage 1=fetch, 2=script, 3=media")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    # Stage control
    if args.stage == 1:
        news = stage_fetch_news(limit=args.limit)
    elif args.stage == 2:
        # load previous fetched items
        news = json.loads((CONTENT_DIR / "fetched_items.json").read_text())
        news = stage_build_scripts(news)
        # save for next stage
        (CONTENT_DIR / "fetched_items.json").write_text(json.dumps(news, indent=2))
    elif args.stage == 3:
        news = json.loads((CONTENT_DIR / "fetched_items.json").read_text())
        stage_create_media(news)
