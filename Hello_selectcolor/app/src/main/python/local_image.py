import os
import cv2

def read_local_image():
    # 在 Chaquopy 中，__file__ 會指向 Python 腳本所在的目錄
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 根據需求，讀取 python_test_image.jpg
    image_path = os.path.join(base_dir, "python_test_image.jpg")
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image")
    
    return buffer.tobytes()
