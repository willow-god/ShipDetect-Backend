import base64
import cv2
import numpy as np

def encode_ndarray_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
