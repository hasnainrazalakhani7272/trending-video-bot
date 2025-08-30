import os
import requests

GRAPH_URL = "https://graph.facebook.com/v20.0"
ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")


class FacebookPoster:
    def __init__(self, access_token):
        self.base_url = GRAPH_URL
        self.user_access_token = access_token
        self.page_access_token = None
        self.page_id = None
        self._setup_token()

    def _check_token_type(self):
        """Check if token is a page token or user token"""
        url = f"{self.base_url}/me"
        params = {"access_token": self.user_access_token}
        resp = requests.get(url, params=params).json()

        if "category" in resp:  # Pages have "category"
            return {"type": "page", "id": resp.get("id"), "name": resp.get("name")}
        elif "id" in resp:
            return {"type": "user", "id": resp.get("id"), "name": resp.get("name")}
        return None

    def _setup_token(self):
        """Decide if access_token is page or user, and set page_id + page_access_token"""
        token_info = self._check_token_type()

        if not token_info:
            raise Exception("❌ Invalid access token")

        if token_info["type"] == "page":
            # Already a page token
            self.page_access_token = self.user_access_token
            self.page_id = token_info["id"]
            print(f"✅ Using page token for: {token_info['name']}")
        else:
            # User token → fetch managed pages
            pages = self.get_pages()
            if not pages:
                raise Exception("❌ No pages found or insufficient permissions")
            page = pages[0]  # auto-pick first page
            self.page_id = page["id"]
            self.page_access_token = page["access_token"]
            print(f"✅ Auto-selected page: {page['name']} (ID: {self.page_id})")

    def get_pages(self):
        """Get pages managed by the user"""
        url = f"{self.base_url}/me/accounts"
        params = {
            "access_token": self.user_access_token,
            "fields": "id,name,access_token,category,tasks"
        }
        resp = requests.get(url, params=params).json()
        return resp.get("data", [])

    def upload_video(self, video_path, caption):
        """Upload video with caption"""
        url = f"{self.base_url}/{self.page_id}/videos"
        files = {"source": open(video_path, "rb")}
        data = {"description": caption, "access_token": self.page_access_token}

        resp = requests.post(url, files=files, data=data).json()
        if "id" in resp:
            print(f"✅ Uploaded {video_path} to page {self.page_id} (Video ID: {resp['id']})")
        else:
            print(f"❌ Upload failed: {resp}")
        return resp


def upload_video(video_path, caption):
    poster = FacebookPoster(ACCESS_TOKEN)
    return poster.upload_video(video_path, caption)

