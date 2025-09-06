import os
import json
import requests
from gtts import gTTS
from transformers import pipeline
import subprocess

# -------------------- API Keys Setup --------------------
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')

# Ensure necessary directories are created
def ensure_directories():
    # List of directories to ensure exist
    directories = ['images', 'output_videos', 'data']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# ------------------ CORE FUNCTION 1: Fetch Trending Content ------------------
def fetch_trending_content(limit=5):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    
    headlines_full_texts = []
    for a in articles[:limit]:
        headline = a.get("title", "No Title")
        full_text = a.get("content", "")  # Get full article text
        headlines_full_texts.append({
            "headline": headline,
            "full_text": full_text
        })
    
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
                images.append((img_url, img_name))  # return URL and name
        else:
            print(f"Error fetching images: {response.status_code}")
    except Exception as e:
        print(f"Error fetching images: {str(e)}")

    return images

# ------------------ CORE FUNCTION 3: Summarize Full Article with Hugging Face ------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(text):
    # Summarize the full article
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

    # Generate TTS narration (audio)
    voice_path = os.path.join(output_dir, f"voice_{index}.mp3")
    gTTS(script, lang="en").save(voice_path)

    # Calculate how long each image should be shown based on desired video duration
    image_display_time = video_duration / len(images)  # Split time evenly between images

    input_images = []
    for img_url, img_name in images:
        img_path = os.path.join(output_dir, img_name)

        # Download and save the image
        if not os.path.exists(img_path):
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as f:
                f.write(img_data)

        # Add image to the ffmpeg input list with adjusted duration
        input_images.extend(["-loop", "1", "-t", str(image_display_time), "-i", img_path])  # Adjust the time per image

    # Build ffmpeg command
    video_path = os.path.join(output_dir, f"news_{index}.mp4")
    filter_complex = "".join([f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];" for i in range(len(images))])
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

# ------------------ NEW FUNCTION 1: Fetch and Save Headlines and Full Texts ------------------
def fetch_and_save_headlines_and_texts(limit=5, save_data=True, headlines_file="data/headlines.json"):
    headlines_full_texts = fetch_trending_content(limit)

    if save_data:
        with open(headlines_file, "w") as f:
            json.dump(headlines_full_texts, f, indent=4)
    
    return headlines_full_texts

# ------------------ NEW FUNCTION 2: Fetch Images and Save ------------------
def fetch_images_and_save(headlines_file="data/headlines.json", images_dir="images", save_data=True, images_file="data/images.json"):
    # Read headlines and full texts from the headlines_file (which now contains both)
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    images_data = []
    
    os.makedirs(images_dir, exist_ok=True)

    # Loop through each article (headline + full_text)
    for article in articles:
        headline = article.get("headline")
        
        # Fetch related images based on the headline
        images = get_related_images(headline, count=3)
        
        # Save the actual images to the specified directory
        saved_images = []
        for img_url, img_name in images:
            img_path = os.path.join(images_dir, img_name)
            # Download and save the image if it doesn't already exist
            if not os.path.exists(img_path):
                try:
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    saved_images.append((img_url, img_name))  # Save both the URL and the name
                except Exception as e:
                    print(f"Error saving image {img_name}: {str(e)}")

        # Store the images data with the corresponding headline
        images_data.append({
            "headline": headline,
            "images": saved_images  # List of tuples: (URL, filename)
        })

        # Optionally save images data (image names and paths) to JSON
        if save_data:
            with open(images_file, "w") as f:
                json.dump(images_data, f, indent=4)

    return images_data

# ------------------ NEW FUNCTION 3: Generate Summaries and Save ------------------
def generate_summaries_and_save(headlines_file="data/headlines.json", summaries_file="data/summaries.json", save_data=True):
    # Read headlines and full texts from the headlines_file (which now contains both)
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    summaries = {}

    # Generate summary for each article
    for article in articles:
        headline = article.get("headline")
        full_text = article.get("full_text")
        if full_text:
            summary = summarize_article(full_text)  # Generate the summary of the full text
            summaries[headline] = summary  # Store the summary with the headline as the key

    # Optionally save summaries data
    if save_data:
        with open(summaries_file, "w") as f:
            json.dump(summaries, f, indent=4)

    return summaries

# ------------------ NEW FUNCTION 4: Create Videos and Save ------------------
def create_videos_and_save(images_file="data/images.json", summaries_file="data/summaries.json", output_dir="output_videos", save_data=True):
    os.makedirs(output_dir, exist_ok=True)

    # Read images and summaries from the files
    with open(images_file, "r") as f:
        images_data = json.load(f)

    with open(summaries_file, "r") as f:
        summaries = json.load(f)

    video_results = []

    for data in images_data:
        headline = data["headline"]
        images = data["images"]
        summary = summaries.get(headline, "")

        # Generate caption and create video
        caption = generate_caption(headline, summary)
        video_path = create_video(images, summary, len(video_results)+1, output_dir)

        video_results.append({
            "headline": headline,
            "video_path": video_path,
            "caption": caption
        })

        # Optionally save the video results
        if save_data:
            with open("data/video_results.json", "w") as f:
                json.dump(video_results, f, indent=4)

    return video_results


# Main Execution
ensure_directories()

# Fetch and save headlines and full article texts
fetch_and_save_headlines_and_texts(limit=1)

# Fetch images and save them
fetch_images_and_save()

# Generate summaries of full articles and save them
generate_summaries_and_save()

# Create videos and save the results
create_videos_and_save()
