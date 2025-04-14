# result_routes.py

import os
import hashlib
import mysql.connector
from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
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
    category_id: int
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
        "category_id": row[2],
        "ship_id": row[3],
        "bbox": row[4],
        "region_url": row[5],
        "timestamp": row[6],
        "confidence": row[7],
        "created_at": row[8].strftime("%Y-%m-%d %H:%M:%S")
    }

@router.get("/get_results", response_model=List[Result])
async def get_results(
    video_ids: Optional[str] = None,
    ship_id: Optional[str] = None,
    category_ids: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
):
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    values = []

    # 处理video_ids参数（逗号分隔转换为列表）
    if video_ids:
        video_id_list = [int(vid.strip()) for vid in video_ids.split(",") if vid.strip()]
        if video_id_list:
            conditions.append(f"video_id IN ({','.join(['%s'] * len(video_id_list))})")
            values.extend(video_id_list)

    # 处理ship_ids参数（逗号分隔转换为列表）
    if ship_id:
        conditions.append("ship_id LIKE %s")
        values.append(f"%{ship_id}%")

    # 处理category_ids参数（逗号分隔转换为列表）
    if category_ids:
        category_id_list = [int(cid.strip()) for cid in category_ids.split(",") if cid.strip()]
        if category_id_list:
            conditions.append(f"category IN ({','.join(['%s'] * len(category_id_list))})")
            values.extend(category_id_list)

    where_clause = " AND ".join(conditions) if conditions else ""
    
    sql = f"""
        SELECT video_id, frame_id, category, ship_id, bbox, region_url, timestamp, confidence, created_at
        FROM results
        {f"WHERE {where_clause}" if where_clause else ""}
        ORDER BY created_at DESC
        LIMIT %s
    """
    values.append(limit)

    cursor.execute(sql, values)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [parse_result_row(row) for row in rows]

# Add this to your result_routes.py

class DailyPassCount(BaseModel):
    date: str
    count: int

class CategoryCount(BaseModel):
    category: str
    count: int

class DataOverview(BaseModel):
    total_week: int
    total_month: int
    total_year: int
    avg_confidence: float
    daily_pass_counts: List[DailyPassCount]
    category_counts: List[CategoryCount]

@router.get("/get_all_datas", response_model=DataOverview)
async def get_all_datas():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get current date for calculations
    current_date = datetime.now().date()
    
    # Calculate date ranges
    week_ago = current_date - timedelta(days=7)
    month_ago = current_date - timedelta(days=30)
    year_ago = current_date - timedelta(days=365)
    
    # 1. Get total counts for different time periods
    cursor.execute("""
        SELECT 
            COUNT(*) as total_week 
        FROM results 
        WHERE created_at >= %s
    """, (week_ago,))
    total_week = cursor.fetchone()['total_week']
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_month 
        FROM results 
        WHERE created_at >= %s
    """, (month_ago,))
    total_month = cursor.fetchone()['total_month']
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_year 
        FROM results 
        WHERE created_at >= %s
    """, (year_ago,))
    total_year = cursor.fetchone()['total_year']
    
    # 2. Get average confidence
    cursor.execute("""
        SELECT 
            AVG(confidence) as avg_confidence 
        FROM results
    """)
    avg_confidence = round(cursor.fetchone()['avg_confidence'], 5)
    
    # 3. Get daily pass counts for last 7 days
    cursor.execute("""
        SELECT 
            DATE(created_at) as date, 
            COUNT(*) as count 
        FROM results 
        WHERE created_at >= %s
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT 7
    """, (week_ago,))
    daily_pass_counts = [
        {"date": row['date'].strftime("%Y-%m-%d"), "count": row['count']} 
        for row in cursor.fetchall()
    ]
    
    # Ensure we have exactly 7 days (fill missing days with 0)
    dates = [(current_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    dates.reverse()
    
    daily_data = {item['date']: item['count'] for item in daily_pass_counts}
    complete_daily_counts = [
        {"date": date, "count": daily_data.get(date, 0)}
        for date in dates
    ]
    
    # 4. Get category counts
    cursor.execute("""
        SELECT 
            category, 
            COUNT(*) as count 
        FROM results 
        GROUP BY category
        ORDER BY count DESC
    """)
    category_counts = [
        {"category": CATEGORY_MAP.get(row['category'], "unknown"), "count": row['count']}
        for row in cursor.fetchall()
    ]
    
    cursor.close()
    conn.close()
    
    return {
        "total_week": total_week,
        "total_month": total_month,
        "total_year": total_year,
        "avg_confidence": avg_confidence,
        "daily_pass_counts": complete_daily_counts,
        "category_counts": category_counts
    }


# 初始化数据库
init_result_table()
