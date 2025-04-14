from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import mysql.connector
import random
from difflib import SequenceMatcher
import os
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

# ---------- Pydantic 模型 ----------
class ShipProfileBase(BaseModel):
    category_id: int
    category_name: str
    ship_id: str

class ShipProfileCreate(BaseModel):
    category_id: int
    ship_id: str

class ShipProfileUpdate(BaseModel):
    category_id: Optional[int] = None
    ship_id: Optional[str] = None

class ShipProfileOut(ShipProfileBase):
    id: int
    created_at: str

# ---------- 数据库连接 ----------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=os.getenv("MYSQL_PORT"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB_NAME")
    )

# ---------- 初始化数据库表 ----------
def init_ship_profile_table():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ship_profiles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            category_id TINYINT NOT NULL,
            category_name VARCHAR(100),
            ship_id VARCHAR(100) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
    """)
    conn.commit()

    # 如果没有数据，插入测试数据
    cursor.execute("SELECT COUNT(*) FROM ship_profiles")
    count = cursor.fetchone()[0]

    if count == 0:
        test_data = []
        for i in range(2):
            cid = random.randint(1, 6)
            cname = CATEGORY_MAP.get(cid, "unknown")
            sid = f"TEST-{random.randint(10000, 99999)}"
            test_data.append((cid, cname, sid))

        cursor.executemany("""
            INSERT INTO ship_profiles (category_id, category_name, ship_id)
            VALUES (%s, %s, %s)
        """, test_data)
        conn.commit()
        print(f"✅ ship_profiles 初始化成功，插入了 {len(test_data)} 条测试数据")

    cursor.close()
    conn.close()

# ---------- CRUD 接口 ----------

@router.get("/ship_profiles", response_model=List[ShipProfileOut])
async def list_ship_profiles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, category_id, category_name, ship_id, created_at FROM ship_profiles ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{
        "id": row[0],
        "category_id": row[1],
        "category_name": row[2],
        "ship_id": row[3],
        "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
    } for row in rows]

@router.post("/ship_profiles", response_model=ShipProfileOut)
async def create_ship_profile(data: ShipProfileCreate):
    category_name = CATEGORY_MAP.get(data.category_id, "unknown")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO ship_profiles (category_id, category_name, ship_id)
            VALUES (%s, %s, %s)
        """, (data.category_id, category_name, data.ship_id))
        conn.commit()
        id = cursor.lastrowid
    except mysql.connector.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Ship ID already exists")
    finally:
        cursor.close()
        conn.close()

    return {
        "id": id,
        "category_id": data.category_id,
        "category_name": category_name,
        "ship_id": data.ship_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@router.put("/ship_profiles/{id}", response_model=ShipProfileOut)
async def update_ship_profile(id: int, update: ShipProfileUpdate):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ship_profiles WHERE id = %s", (id,))
    row = cursor.fetchone()

    if not row:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")

    new_category_id = update.category_id if update.category_id is not None else row[1]
    new_category_name = CATEGORY_MAP.get(new_category_id, row[2])
    new_ship_id = update.ship_id if update.ship_id is not None else row[3]

    try:
        cursor.execute("""
            UPDATE ship_profiles
            SET category_id=%s, category_name=%s, ship_id=%s
            WHERE id=%s
        """, (new_category_id, new_category_name, new_ship_id, id))
        conn.commit()
    except mysql.connector.IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Duplicate ship ID")
    finally:
        cursor.close()
        conn.close()

    return {
        "id": id,
        "category_id": new_category_id,
        "category_name": new_category_name,
        "ship_id": new_ship_id,
        "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
    }

@router.delete("/ship_profiles/{id}")
async def delete_ship_profile(id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ship_profiles WHERE id = %s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    return {"success": True, "message": f"Ship profile {id} deleted."}

def calculate_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

@router.get("/ship_profiles/search", response_model=List[ShipProfileOut])
async def search_ship_profiles(q: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, category_id, category_name, ship_id, created_at FROM ship_profiles")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    matched = []
    for row in rows:
        similarity = calculate_similarity(q.lower(), row[3].lower())
        if similarity > 0.5:
            matched.append((
                similarity,
                {
                    "id": row[0],
                    "category_id": row[1],
                    "category_name": row[2],
                    "ship_id": row[3],
                    "created_at": row[4].strftime("%Y-%m-%d %H:%M:%S")
                }
            ))

    # 可按相似度排序（可选）
    matched.sort(key=lambda x: x[0], reverse=True)

    return [item[1] for item in matched]

# 接口 /categories 返回一个类别字典，方便前端使用
@router.get("/categories")
async def get_categories():
    return CATEGORY_MAP 

# ---------- 初始化表 ----------
init_ship_profile_table()
