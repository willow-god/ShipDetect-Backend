import asyncio
import os
import tempfile
import cv2
import json
import numpy as np

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.api.yolov8_routes import yolov8_detect
from app.api.ppocr_routes import ppocr_v4
from app.utils.pic2base64 import encode_ndarray_to_base64

router = APIRouter()

@router.post("/test_image")
async def detect_image(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    detections = yolov8_detect(img)
    print(f"检测到 {len(detections)} 个船舶")
    results = []

    for idx, det in enumerate(detections, start=1):
        print(f"船舶 {idx}: {det['category']}")
        
        bbox = list(map(int, det["bbox"]))  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        # ---------- 处理船舶 bbox ----------
        print(f"船舶 bbox: {bbox}")
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bbox: {bbox}")
            continue
        region = img[y1:y2, x1:x2].copy()  # 裁剪区域图像
        print(f"裁剪区域大小: {region.shape}")

        # ---------- 绘制船舶 bbox 在原图上 ----------
        img_with_ship_box = img.copy()
        cv2.rectangle(img_with_ship_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------- 船号识别 ----------
        ocr_result = ppocr_v4(region)
        ship_number = ocr_result.get("ship_id", "")
        number_bbox = ocr_result.get("ship_id_bbox", [])  # 相对于 region 的 bbox

        # ---------- 绘制船号 bbox ----------
        region_with_number_box = region.copy()
        if number_bbox:
            nx1, ny1, nx2, ny2 = map(int, number_bbox)
            cv2.rectangle(region_with_number_box, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)

        # ---------- 编码为 Base64 ----------
        ship_img_b64 = encode_ndarray_to_base64(img_with_ship_box)
        number_crop_img_b64 = encode_ndarray_to_base64(region_with_number_box)

        results.append({
            "id": idx,
            "category": det["category"],
            "ship_bbox": bbox,
            "visualized_ship_image": ship_img_b64,
            "ship_number": ship_number,
            "number_bbox": number_bbox,
            "visualized_number_on_crop": number_crop_img_b64
        })

    return {"results": results}

@router.post("/test_video")
async def stream_video_detect(video: UploadFile = File(...)):
    # 保存上传的视频到临时文件
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await video.read())
    temp.close()

    async def gen():
        cap = None
        try:
            cap = cv2.VideoCapture(temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 25.0
            frame_interval = int(fps)
            frame_id = 0
            max_width = 1280

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % frame_interval != 0:
                    frame_id += 1
                    continue

                timestamp = round(frame_id / fps, 2)
                print(f"Processing frame {frame_id} at timestamp {timestamp:.2f}s")

                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                detections = yolov8_detect(frame)
                frame_drawn = frame.copy()
                result_list = []

                for det in detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    ship_crop = frame[y1:y2, x1:x2].copy()
                    ship_category = det["category"]
                    ship_confidence = round(float(det.get("score", 0.9)), 3)

                    ocr_result = ppocr_v4(ship_crop)
                    ship_id = ocr_result.get("ship_id", "")
                    ship_id_bbox_crop = ocr_result.get("ship_id_bbox", [])
                    ship_id_conf = round(float(ocr_result.get("ship_id_score", 0.85)), 3)

                    if ship_id_bbox_crop:
                        sx1, sy1, sx2, sy2 = map(int, ship_id_bbox_crop)
                        ship_id_bbox_global = [x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2]
                        cv2.rectangle(frame_drawn, (ship_id_bbox_global[0], ship_id_bbox_global[1]),
                                      (ship_id_bbox_global[2], ship_id_bbox_global[3]), (0, 0, 255), 2)
                    else:
                        ship_id_bbox_global = []

                    cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    result_list.append({
                        "category": ship_category,
                        "ship_id": ship_id,
                        "ship_bbox": [x1, y1, x2, y2],
                        "ship_confidence": ship_confidence,
                        "ship_id_bbox": ship_id_bbox_global,
                        "ship_id_confidence": ship_id_conf
                    })

                visualized_b64 = encode_ndarray_to_base64(frame_drawn)

                yield json.dumps({
                    "status": "ok",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "visualized_frame": visualized_b64,
                    "results": result_list
                }) + "\n"
                await asyncio.sleep(0)  # 推动 event loop 输出帧

                frame_id += 1

            # 所有帧处理完毕，发送一个结束信号
            yield json.dumps({
                "status": "done"
            }) + "\n"

        except Exception as e:
            yield json.dumps({
                "status": "error",
                "message": f"视频处理失败：{str(e)}"
            }) + "\n"

        finally:
            if cap:
                cap.release()
            if os.path.exists(temp.name):
                os.remove(temp.name)

    return StreamingResponse(gen(), media_type="application/json", status_code=200)