from pathlib import Path
import requests
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

LSKY_PRO_TOKEN = os.getenv("LSKY_PRO_TOKEN")
LSKY_PRO_URL = os.getenv("LSKY_PRO_URL")

# 模拟上传至兰空图床，返回图片直链
def simulate_upload_to_lsky(file_path):
    return f"https://img.lsky.test/{Path(file_path).name}"

def upload_to_lsky(file_path, strategy_id=None):
    url = f"{LSKY_PRO_URL}/upload"
    headers = {
        "Authorization": LSKY_PRO_TOKEN,
    }
    files = {
        "file": open(file_path, "rb")
    }
    data = {}
    if strategy_id is not None:
        data["strategy_id"] = strategy_id

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()  # 会抛出异常如果状态码不是 2xx

        json_resp = response.json()
        if json_resp.get("status"):
            # 成功上传，返回直链 URL
            return json_resp["data"]["links"]["url"]
        else:
            raise Exception(f"上传失败: {json_resp.get('message')}")

    except Exception as e:
        print(f"上传异常: {e}")
        return None
    
# 测试样例尝试
if __name__ == "__main__":
    # 测试上传
    file_path = "E:/Graduate_Design/ShipDetect-Backend/resources/images/000002.jpg"  # 替换为你的图片路径
    url = upload_to_lsky(file_path)
    if url:
        print(f"上传成功，直链 URL: {url}")
    else:
        print("上传失败")