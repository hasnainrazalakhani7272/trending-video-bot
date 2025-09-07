from just_video import (
    ensure_directories,
    fetch_and_save_headlines_and_texts,
    fetch_images_and_save,
    generate_summaries_and_save,
    create_videos_and_save
)
from new_facebook_video_upload import upload_video

if __name__ == "__main__":
    # Step 1: Generate news videos
    ensure_directories()
    fetch_and_save_headlines_and_texts(limit=1)
    fetch_images_and_save()
    generate_summaries_and_save()
    
    video_results = create_videos_and_save()
    
    # Step 2: Upload each video to Facebook
    if video_results:
        for item in video_results:
            video_path = item.get("video_path")
            caption = item.get("caption")
            if video_path and caption:
                print(f"\n🚀 Uploading {video_path} ...")
                upload_video(video_path, caption)
            else:
                print(f"⚠️ Skipping upload: Missing video or caption in {item}")
    else:
        print("⚠️ No videos generated to upload.")
