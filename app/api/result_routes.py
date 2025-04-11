# result_routes.py

import os
import hashlib
import mysql.connector
from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

CATEGORY_MAP = {
    1: "ore carrier",
    2: "bulk cargo carrier",
    3: "general cargo ship",
    4: "container ship",
    5: "fishing boat",
    6: "passenger ship"
}

# 建立数据库连接
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=os.getenv("MYSQL_PORT"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB_NAME")
    )

# 初始化数据库表
def init_result_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        video_id INT,
        frame_id VARCHAR(20),
        category TINYINT,
        ship_id VARCHAR(100),
        bbox VARCHAR(100),
        region_url VARCHAR(255),
        timestamp VARCHAR(50),
        confidence FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
    """)
    conn.commit()
    cursor.close()
    conn.close()

# 写入单条检测结果（供后端处理调用）
def save_result_to_db(video_id: int, ship_id: str, bbox: str, region_url: str,
                      timestamp: str, category: int, confidence: float):
    hash_input = f"{region_url}_{datetime.now().timestamp()}"
    frame_id = "fid_" + hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results (video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence))
    conn.commit()
    cursor.close()
    conn.close()

# 查询返回结构
class Result(BaseModel):
    video_id: int
    frame_id: str
    category: str
    ship_id: str
    bbox: str
    region_url: str
    timestamp: str
    confidence: float
    created_at: str

def parse_result_row(row):
    return {
        "video_id": row[0],
        "frame_id": row[1],
        "category": CATEGORY_MAP.get(row[2], "unknown"),
        "ship_id": row[3],
        "bbox": row[4],
        "region_url": row[5],
        "timestamp": row[6],
        "confidence": row[7],
        "created_at": row[8].strftime("%Y-%m-%d %H:%M:%S")
    }

# 查询接口

@router.get("/get_results_by_video_id", response_model=List[Result])
async def get_results_by_video_id(video_id: int, limit: int = Query(default=100, le=1000)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence, created_at
        FROM results WHERE video_id = %s ORDER BY created_at DESC LIMIT %s
    """, (video_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [parse_result_row(row) for row in rows]

@router.get("/get_results_by_ship_id", response_model=List[Result])
async def get_results_by_ship_id(ship_id: str, limit: int = Query(default=100, le=1000)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence, created_at
        FROM results WHERE ship_id = %s ORDER BY created_at DESC LIMIT %s
    """, (ship_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [parse_result_row(row) for row in rows]

@router.get("/get_results_by_category", response_model=List[Result])
async def get_results_by_category(category: int = Query(..., ge=1, le=6), limit: int = Query(default=100, le=1000)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence, created_at
        FROM results WHERE category = %s ORDER BY created_at DESC LIMIT %s
    """, (category, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [parse_result_row(row) for row in rows]

@router.get("/get_all_results", response_model=List[Result])
async def get_all_results(limit: int = Query(default=100, le=1000)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence, created_at
        FROM results ORDER BY created_at DESC LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [parse_result_row(row) for row in rows]

# 初始化数据库
init_result_table()
