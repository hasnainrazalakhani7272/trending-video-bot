from news_to_videos import create_videos_from_news
from facebook_poster import upload_video

if __name__ == "__main__":
    # Step 1: Generate news videos
    videos = create_videos_from_news(limit=2)

    # Step 2: Upload each to Facebook
    for item in videos:
        print(f"\n🚀 Uploading {item['video_path']} ...")
        upload_video(item["video_path"], item["caption"])

