from just_video import ensure_directories
from just_video import fetch_and_save_headlines_and_texts
from just_video import fetch_images_and_save
from just_video import generate_summaries_and_save
from just_video import create_videos_and_save
from facebook_poster import upload_video

if __name__ == "__main__":
    
    # Step 1: Generate news videos
    ensure_directories()
    fetch_and_save_headlines_and_texts(limit=1)
    fetch_images_and_save()
    generate_summaries_and_save()
    result = create_videos_and_save()
    # Step 2: Upload each to Facebook
    for item in result:
        print(f"\n🚀 Uploading {item['video_path']} ...")
        upload_video(item["video_path"], item["caption"])
