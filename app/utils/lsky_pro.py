from pathlib import Path

# 模拟上传至兰空图床，返回图片直链
def simulate_upload_to_lsky(file_path):
    return f"https://img.lsky.test/{Path(file_path).name}"