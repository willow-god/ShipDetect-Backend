# video_routes.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import threading
import mysql.connector
from datetime import datetime
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv
import requests
from .result_routes import save_result_to_db
from .yolov8_routes import simulate_yolov8_detect, yolov8_detect
from .ppocr_routes import simulate_ppocr, ppocr_v4
from app.utils.lsky_pro import upload_to_lsky
import tempfile
import uuid
import cv2

router = APIRouter()

# 加载环境变量
load_dotenv()

# 状态枚举值
STATUS_PROCESSING = 1
STATUS_COMPLETED = 2
STATUS_FAILED = 3

# 连接 MySQL 数据库
def get_db_connection():
    db_config = {
        'host': os.getenv("MYSQL_HOST"),
        'port': os.getenv("MYSQL_PORT"),
        'user': os.getenv("MYSQL_USER"),
        'password': os.getenv("MYSQL_PASSWORD"),
        'database': os.getenv("MYSQL_DB_NAME")
    }
    conn = mysql.connector.connect(**db_config)
    return conn

# 初始化数据库，创建视频表和添加示例数据
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INT AUTO_INCREMENT PRIMARY KEY,
        video_name VARCHAR(255) NOT NULL,
        video_url VARCHAR(255),
        status TINYINT DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
    """)

    # 添加示例数据（每条语句分开执行）
    example_data = [
        ('视频1', 'http://example.com/video1.mp4', STATUS_PROCESSING),
        ('视频2', 'http://example.com/video2.m3u8', STATUS_COMPLETED),
        ('视频3', 'http://example.com/video3.mp4', STATUS_FAILED)
    ]
    for name, url, status in example_data:
        cursor.execute("""
            INSERT INTO videos (video_name, video_url, status)
            SELECT * FROM (SELECT %s, %s, %s) AS tmp
            WHERE NOT EXISTS (
                SELECT video_name FROM videos WHERE video_name = %s
            ) LIMIT 1;
        """, (name, url, status, name))

    conn.commit()
    cursor.close()
    conn.close()

# 状态数字转文本
def status_to_text(status: int) -> str:
    if status == STATUS_COMPLETED:
        return "处理完成"
    elif status == STATUS_FAILED:
        return "处理失败"
    return "处理中"

# 保存处理结果到数据库
async def process_video(video_id: int, video_url: str):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 更新视频状态为处理中
        cursor.execute("UPDATE videos SET status = %s WHERE id = %s", (STATUS_PROCESSING, video_id))
        print(f"视频 {video_id} 状态更新为【处理中】")
        conn.commit()

        # 下载或读取视频
        if video_url.startswith("http"):
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, 'wb') as f:
                f.write(requests.get(video_url).content)
        else:
            video_path = video_url

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 清空输出帧目录
        shutil.rmtree("output/frames", ignore_errors=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps == 0:
            fps = 30  # 默认帧率
            print(f"视频 {video_id} FPS 获取失败，使用默认值: {fps}")

        frame_interval = int(fps * 3)  # 每3秒抽一帧
        print(f"视频 {video_id} FPS: {fps}，每 {frame_interval} 帧抽一帧")

        frame_index = 0
        print(f"视频 {video_id} 开始抽帧处理")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"视频 {video_id} 处理完成")
                break

            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

            yolov8_results = yolov8_detect(frame)  # 模拟检测
            print(f"帧 {frame_index} 检测到 {len(yolov8_results)} 个目标")

            for det in yolov8_results:
                x1, y1, x2, y2 = det['bbox']
                region = frame[y1:y2, x1:x2]
                region_path = f"output/frames/{uuid.uuid4().hex}.jpg"
                os.makedirs(os.path.dirname(region_path), exist_ok=True)
                cv2.imwrite(region_path, region)

                print(f"开始对帧 {frame_index} 的目标进行 OCR")
                ocr_results = ppocr_v4(region_path)
                ship_id = ocr_results['ship_id']
                ship_id_bbox = ocr_results['ship_id_bbox']

                # 上传图床
                ship_id_url = upload_to_lsky(region_path)

                # 保存数据库
                save_result_to_db(
                    video_id=video_id,
                    ship_id=ship_id,
                    bbox=str(ship_id_bbox),
                    region_url=ship_id_url,
                    timestamp=timestamp_str,
                    category=det['category_id'],
                    confidence=det['confidence']
                )

                print(f"帧 {frame_index} OCR 完成，识别到船牌号: {ship_id}")

            frame_index += 1

        # 更新视频状态为完成
        cursor.execute("UPDATE videos SET status = %s WHERE id = %s", (STATUS_COMPLETED, video_id))
        conn.commit()

    except Exception as e:
        print(f"视频 {video_id} 处理失败，错误：{str(e)}")
        cursor.execute("UPDATE videos SET status = %s WHERE id = %s", (STATUS_FAILED, video_id))
        conn.commit()

    finally:
        cursor.close()
        conn.close()


# 新增：将 process_video 包装成一个普通的同步函数
def run_process_video(video_id, video_url):
    asyncio.run(process_video(video_id, video_url))

# 定义请求的 schema
class Video(BaseModel):
    video_name: str
    video_url: str

class VideoResponse(BaseModel):
    id: int
    video_name: str
    video_url: str
    status: str
    created_at: str

# 初始化数据库
init_db()

# 视频添加接口
@router.post("/add_video", response_model=VideoResponse)
async def add_video(video: Video):

    conn = get_db_connection()
    cursor = conn.cursor()

    # 插入视频数据
    cursor.execute("INSERT INTO videos (video_name, video_url) VALUES (%s, %s)", 
                  (video.video_name, video.video_url))
    conn.commit()

    # 获取刚插入的 video_id
    video_id = cursor.lastrowid

    cursor.close()
    conn.close()

    # 异步模拟视频处理
    print(f"开始处理视频: {video_id} ({video.video_url})")
    thread = threading.Thread(target=run_process_video, args=(video_id, video.video_url))
    thread.start()

    return {
        "id": video_id,
        "video_name": video.video_name,
        "video_url": video.video_url,
        "status": "处理中",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@router.post("/upload_video", response_model=VideoResponse)
async def upload_video(file: UploadFile = File(...), video_name: str = Form(...)):
    # 创建存储目录
    save_dir = "output/videos"
    # 生成绝对路径
    save_dir = os.path.abspath(save_dir)
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 构造文件名和保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_{timestamp}.mp4"
    save_path = os.path.join(save_dir, filename)

    # 保存文件到本地
    with open(save_path, "wb") as buffer:
        buffer.write(await file.read())

    # 数据库记录
    conn = get_db_connection()
    cursor = conn.cursor()
    relative_path = save_path
    cursor.execute("INSERT INTO videos (video_name, video_url) VALUES (%s, %s)", 
                  (video_name, filename))
    conn.commit()
    video_id = cursor.lastrowid
    cursor.close()
    conn.close()

    # 异步处理
    print(f"开始处理视频: {video_name} ({relative_path})")
    thread = threading.Thread(target=run_process_video, args=(video_id, relative_path))
    thread.start()

    return {
        "id": video_id,
        "video_name": video_name,
        "video_url": relative_path,
        "status": "处理中",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# 视频删除接口
@router.delete("/delete_video/{video_id}")
async def delete_video(video_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 删除视频数据
    cursor.execute("DELETE FROM videos WHERE id = %s", (video_id,))
    conn.commit()

    cursor.close()
    conn.close()

    return {"message": f"Video with ID {video_id} has been deleted successfully."}

# 视频查询接口
@router.get("/get_all_videos", response_model=List[VideoResponse])
async def get_all_videos():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询所有视频
    cursor.execute("SELECT id, video_name, video_url, status, created_at FROM videos")
    rows = cursor.fetchall()

    videos = []
    for row in rows:
        videos.append({
            "id": row[0],
            "video_name": row[1],
            "video_url": row[2],
            "status": status_to_text(row[3]),
            "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
        })

    cursor.close()
    conn.close()

    return videos

# 实现接口，获取所有视频的 ID 列表，格式为 number[]
@router.get("/get_video_ids", response_model=List[int])
async def get_video_ids():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 查询所有视频 ID
    cursor.execute("SELECT id FROM videos")
    rows = cursor.fetchall()

    video_ids = [row[0] for row in rows]

    cursor.close()
    conn.close()

    return video_ids