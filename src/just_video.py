import os
import json
import requests
from gtts import gTTS
from transformers import pipeline
import subprocess

# -------------------- API Keys Setup --------------------
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')

# -------------------- Text2Text Query Generator --------------------
query_generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_image_search_query(headline):
    prompt = f"Generate keywords for image search: {headline}"
    result = query_generator(prompt, max_length=20, do_sample=False)
    query = result[0]['generated_text']
    return query

# Ensure necessary directories are created
def ensure_directories():
    directories = ['images', 'output_videos', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# ------------------ CORE FUNCTION 1: Fetch Trending Content ------------------
def fetch_trending_content(limit=5, fetch_more=20):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])

    # Take more articles than limit to increase variety
    articles = articles[:fetch_more]

    # Extract headline/full_text pairs
    headlines_full_texts = [{
        "headline": a.get("title", "No Title"),
        "full_text": a.get("content", "")
    } for a in articles]

    # Randomly sample 'limit' articles to increase variation
    if len(headlines_full_texts) > limit:
        headlines_full_texts = random.sample(headlines_full_texts, limit)

    return headlines_full_texts

# ------------------ CORE FUNCTION 2: Fetch Related Images from Pexels ------------------
def get_related_images(query, count=3):
    images = []
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={count}'
    headers = {'Authorization': PEXELS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for i, img in enumerate(data['photos']):
                img_url = img['src']['original']
                img_name = f"{query.replace(' ', '_')}_{i}.jpg"
                images.append((img_url, img_name))
        else:
            print(f"Error fetching images: {response.status_code}")
    except Exception as e:
        print(f"Error fetching images: {str(e)}")

    return images

# ------------------ CORE FUNCTION 3: Summarize Full Article with Hugging Face ------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(text):
    summary = summarizer(text, max_length=200, min_length=100, do_sample=False)
    return summary[0]['summary_text']

# ------------------ CORE FUNCTION 4: Generate Social Caption ------------------
def generate_caption(headline, summary):
    hashtags = ["#news", "#update", "#video", "#breakingnews", "#currentevents", "#newsfeed"]
    if "technology" in headline.lower():
        hashtags.extend(["#technews", "#technology", "#innovation"])
    elif "sports" in headline.lower():
        hashtags.extend(["#sports", "#sportsnews", "#athletics"])
    elif "finance" in headline.lower():
        hashtags.extend(["#finance", "#economy", "#stocks", "#cryptocurrency"])
    hashtag_str = " ".join(hashtags)
    caption = f"Catch up on the latest news!\n{headline}\n{summary}\n{hashtag_str}"
    return caption

# ------------------ CORE FUNCTION 5: Create Video ------------------
def create_video(images, script, index, output_dir, video_duration=60):
    os.makedirs(output_dir, exist_ok=True)
    voice_path = os.path.join(output_dir, f"voice_{index}.mp3")
    gTTS(script, lang="en").save(voice_path)

    image_display_time = video_duration / len(images)
    input_images = []
    for img_url, img_name in images:
        img_path = os.path.join(output_dir, img_name)
        if not os.path.exists(img_path):
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as f:
                f.write(img_data)
        input_images.extend(["-loop", "1", "-t", str(image_display_time), "-i", img_path])

    video_path = os.path.join(output_dir, f"news_{index}.mp4")
    filter_complex = "".join([
        f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
        for i in range(len(images))
    ])
    filter_complex += "".join([f"[v{i}]" for i in range(len(images))])
    filter_complex += f"concat=n={len(images)}:v=1:a=0,format=yuv420p[v]"

    cmd = [
        "ffmpeg", "-y",
        *input_images,
        "-i", voice_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", f"{len(images)}:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", video_path
    ]

    subprocess.run(cmd, check=True)
    return video_path

# ------------------ Fetch and Save Headlines and Full Texts ------------------
def fetch_and_save_headlines_and_texts(limit=5, save_data=True, headlines_file="data/headlines.json"):
    headlines_full_texts = fetch_trending_content(limit)
    if save_data:
        with open(headlines_file, "w") as f:
            json.dump(headlines_full_texts, f, indent=4)
    return headlines_full_texts

# ------------------ Fetch Images and Save (with Fallback Strategy) ------------------
def fetch_images_and_save(headlines_file="data/headlines.json", images_dir="images", save_data=True, images_file="data/images.json"):
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    images_data = []
    os.makedirs(images_dir, exist_ok=True)

    for article in articles:
        headline = article.get("headline")
        images = get_related_images(headline, count=3)

        # If no images, try generating an alternative query
        if not images:
            print(f"No images found for headline: '{headline}'")
            alt_query = generate_image_search_query(headline)
            print(f"Retrying with generated query: '{alt_query}'")
            images = get_related_images(alt_query, count=3)

        # Final fallback to generic "news" if still no images
        if not images:
            print("Using fallback query: 'news'")
            images = get_related_images("news", count=3)

        saved_images = []
        for img_url, img_name in images:
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                try:
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    saved_images.append((img_url, img_name))
                except Exception as e:
                    print(f"Error saving image {img_name}: {str(e)}")

        images_data.append({
            "headline": headline,
            "images": saved_images
        })

        if save_data:
            with open(images_file, "w") as f:
                json.dump(images_data, f, indent=4)

    return images_data

# ------------------ Generate Summaries and Save ------------------
def generate_summaries_and_save(headlines_file="data/headlines.json", summaries_file="data/summaries.json", save_data=True):
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    summaries = {}
    for article in articles:
        headline = article.get("headline")
        full_text = article.get("full_text")
        if full_text:
            summary = summarize_article(full_text)
            summaries[headline] = summary

    if save_data:
        with open(summaries_file, "w") as f:
            json.dump(summaries, f, indent=4)

    return summaries

# ------------------ Create Videos and Save ------------------
def create_videos_and_save(images_file="data/images.json", summaries_file="data/summaries.json", output_dir="output_videos", save_data=True):
    os.makedirs(output_dir, exist_ok=True)

    with open(images_file, "r") as f:
        images_data = json.load(f)

    with open(summaries_file, "r") as f:
        summaries = json.load(f)

    video_results = []

    for data in images_data:
        headline = data["headline"]
        images = data["images"]
        summary = summaries.get(headline, "")

        caption = generate_caption(headline, summary)
        video_path = create_video(images, summary, len(video_results)+1, output_dir)

        video_results.append({
            "headline": headline,
            "video_path": video_path,
            "caption": caption
        })

        if save_data:
            with open("data/video_results.json", "w") as f:
                json.dump(video_results, f, indent=4)

    return video_results

# ------------------ Main Execution ------------------
ensure_directories()
fetch_and_save_headlines_and_texts(limit=1)
fetch_images_and_save()
generate_summaries_and_save()
create_videos_and_save()
