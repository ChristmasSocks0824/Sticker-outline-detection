import cv2
import numpy as np
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# --- 常數與設定 ---
MATCH_THRESHOLD = 0.30
TEMPLATE_PATH = 'shapes208.png'
BASE_SIZE = 512

template_cache : Dict[str, Any] = {}
_KERNEL = np.ones((5,5), dtype=np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=(8,8))
def auto_canny(image, sigma=0.33):
    if image is None: return None
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, high)

def get_contours(binary: np.ndarray, noise_floor: int) -> List[np.ndarray]:
    """
    獲取影像中的有效輪廓，並採用「動態自適應面積」進行篩選。
    noise_floor: 接收來自 UI 的 Area 參數，作為絕對過濾底線。
    """
    def _keep(contours):
        all_areas = [cv2.contourArea(c) for c in contours]
        
        valid_contours = []
        valid_areas = []

        for c, area in zip(contours, all_areas):
            # 【關鍵連線】使用來自 UI 的動態 noise_floor
            if area > noise_floor:
                valid_contours.append(c)
                valid_areas.append(area)

        if not valid_areas:
            return []

        # 核心：動態基準 - 以真實物件中的「最小面積」打 9 折作為過濾門檻
        dynamic_min_area = min(valid_areas)
        auto_threshold = dynamic_min_area * 0.9

        kept = []
        for c, area in zip(valid_contours, valid_areas):
            if area >= auto_threshold :
                kept.append(c)

        # 依面積由大到小排序
        kept.sort(key=cv2.contourArea, reverse=True)
        return kept

    # 1. 關鍵修正：改用 RETR_LIST 而非 RETR_EXTERNAL
    # 理由：您的影像中物體 (貼紙) 位在另一個閉合輪廓 (手機殼) 內部。
    # RETR_EXTERNAL 只會回傳最外層的閉合曲線，這就是為什麼您只能看到手機殼的原因。
    # 改用 RETR_LIST 可以忽略層級關係，強制掃描出所有獨立的閉合區域。
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = _keep(contours)
    
    if not result:
        # 如果 LIST 也沒結果 (通常不可能)，維持安全機制
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = _keep(contours)
    return result

def match_score(template_contour: np.ndarray, contour: np.ndarray) -> float:
    s = [cv2.matchShapes(template_contour, contour, method, 0.0)
         for method in (cv2.CONTOURS_MATCH_I1, cv2.CONTOURS_MATCH_I2, cv2.CONTOURS_MATCH_I3)]
    return float(min(s))

def get_angle(m: Dict[str, float]) -> float:
    mu20, mu02, mu11 = m.get('mu20', 0), m.get('mu02', 0), m.get('mu11', 0)
    if (mu20 - mu02) == 0: return 0.0
    return math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))

def get_template_contour(template_name: str) -> np.ndarray:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, template_name)
    template_img = cv2.imread(path)
    if template_img is None: raise ValueError(f'Template failed: {path}')
    template_img = cv2.resize(template_img, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blur)
    dilate = cv2.dilate(edges, _KERNEL, iterations=1)
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: raise ValueError('No contours in template')
    return max(cnts, key=cv2.contourArea)

def canny_from_image_bytes(image_data: bytes, mode: str = "Contour Detection", gamma_val: float = 0.7, noise_floor: int = 300) -> Dict[str, Any]:
    """
    主要處理入口。
    noise_floor: 由 Java 端透過 UI 設定傳入，實現即時更新。
    """
    result_data = {"image": None, "angle": 0.0, "status": "OK", "error_msg": ""}
    try:
        raw_buffer = np.frombuffer(image_data, dtype=np.uint8)
        source_BGR = cv2.imdecode(raw_buffer, cv2.IMREAD_COLOR)
        if source_BGR is None: raise ValueError("Decode failed")
        
        h_orig, w_orig = source_BGR.shape[:2]
        target_512 = cv2.resize(source_BGR, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_AREA)
        ratio = h_orig / float(BASE_SIZE)

        gray = cv2.cvtColor(target_512, cv2.COLOR_BGR2GRAY)

        # 影像增強
        lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
        gray_gamma = cv2.LUT(gray, lut)
        target_gamma = _CLAHE.apply(gray_gamma)
        blur1 = cv2.GaussianBlur(target_gamma, (5, 5), 1.0)
        mask = cv2.subtract(gray_gamma, blur1)
        k = 1.5
        sharpened = cv2.addWeighted(gray_gamma, 1.0, mask, k, 0)
        blur_final = cv2.GaussianBlur(sharpened, (5, 5), 0)
        _, otsu = cv2.threshold(blur_final, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # MORPH_CLOSE (閉運算)：先將 IC 外部接腳與主體之間的細微縫隙填滿，擴張後收縮，確保輪廓完整
        cleaned_binary = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
        # MORPH_OPEN (開運算)：再去除背景中可能因為雜訊產生的微小白色斑點（修正了原本變數覆蓋的 Bug）
        cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, _KERNEL, iterations=1)
        edges = auto_canny(cleaned_binary)
        dilate = cv2.dilate(edges, _KERNEL, iterations=1)

        if mode == "Debug Pre-processing":
            filtered_img = cv2.bilateralFilter(target_512, d=9, sigmaColor=75, sigmaSpace=75)
            # 步驟一：轉換為 HSV 空間，並偵測柴犬貼紙的特定黃褐色/橙色與
            # 1. 轉為 HSV 色彩空間
            # 步驟一：轉換為 HSV 空間，並偵測柴犬貼紙的特定黃褐色/橙色與膚色調
            hsv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
            # ================= 新增：提取 V 通道做 CLAHE，再拼回 HSV =================
            # 1. 拆分 HSV 三個通道
            h, s, v = cv2.split(hsv)

            # 2. 對 V (亮度) 通道應用 CLAHE，消除手機殼表面光影與反射不均
            # 這裡直接套用你原本定義好的 全域 _CLAHE 物件
            v_clahe = _CLAHE.apply(v)

            # 3. 將處理後的 v_clahe 與原本的 h, s 通道重新合併回新的 HSV 影像
            hsv = cv2.merge([h, s, v_clahe])

            # 2. 建立遮罩
            lower_shiba = np.array([0, 32, 43])
            upper_shiba = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_shiba, upper_shiba)

            lower_white = np.array([0, 18, 150])
            upper_white = np.array([180, 55, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # 合併兩者得到初步的 shiba_mask
            shiba_mask = cv2.bitwise_or(skin_mask, white_mask)

            # =========================================================================
            # 【新增】外輪廓填滿法 (Contour Filling)
            # 目的：直接無視內部黑洞，將貼紙主體一次填實
            # =========================================================================
            filled_mask = shiba_mask.copy()

            # 尋找最外層輪廓
            contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 將所有外部輪廓包圍的區域「一次塗滿白色」
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
            # =========================================================================
            # 後續修飾：移除邊緣噪點
            # =========================================================================
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            # 先做開運算斷開與邊框的黏連
            opened_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
            # 再做微侵蝕讓輪廓更漂亮
            eroded_mask = cv2.erode(opened_mask, kernel_ellipse, iterations=1)

            # 步驟三：使用 bitwise_and 與遮罩合併，提取出「只有柴犬貼紙」且「其餘背景全黑」的超乾淨影像
            target_512_cleaned = cv2.bitwise_and(target_512, target_512, mask=eroded_mask)

            # =========================================================================
            # 將淨化後的 target_512_cleaned 送入你原本的影像增強與二值化流程
            # =========================================================================
            gray = cv2.cvtColor(target_512_cleaned, cv2.COLOR_BGR2GRAY)
            #gamma1 = 2.2
            # 影像增強
            lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
            gray_gamma = cv2.LUT(gray, lut)
            target_gamma_CLAHE_hsv = _CLAHE.apply(gray_gamma)

            debug_view = cv2.resize(target_gamma_CLAHE_hsv , (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode(".png", debug_view)
            result_data["image"] = buf.tobytes()
            return result_data
        
        # 【關鍵傳遞】將 UI 的 noise_floor 傳入核心過濾函式
        target_contours = get_contours(dilate, noise_floor)

        if mode == "HSV_findContour":
            filtered_img = cv2.bilateralFilter(target_512, d=9, sigmaColor=75, sigmaSpace=75)
            # 步驟一：轉換為 HSV 空間，並偵測柴犬貼紙的特定黃褐色/橙色與
            # 1. 轉為 HSV 色彩空間
            # 步驟一：轉換為 HSV 空間，並偵測柴犬貼紙的特定黃褐色/橙色與膚色調
            hsv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
            # ================= 新增：提取 V 通道做 CLAHE，再拼回 HSV =================
            # 1. 拆分 HSV 三個通道
            h, s, v = cv2.split(hsv)

            # 2. 對 V (亮度) 通道應用 CLAHE，消除手機殼表面光影與反射不均
            # 這裡直接套用你原本定義好的 全域 _CLAHE 物件
            v_clahe = _CLAHE.apply(v)

            # 3. 將處理後的 v_clahe 與原本的 h, s 通道重新合併回新的 HSV 影像
            hsv = cv2.merge([h, s, v_clahe])

            # 2. 建立遮罩
            lower_shiba = np.array([0, 32, 43])
            upper_shiba = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_shiba, upper_shiba)

            lower_white = np.array([0, 18, 150])
            upper_white = np.array([180, 55, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # 合併兩者得到初步的 shiba_mask
            shiba_mask = cv2.bitwise_or(skin_mask, white_mask)

            # =========================================================================
            # 【新增】外輪廓填滿法 (Contour Filling)
            # 目的：直接無視內部黑洞，將貼紙主體一次填實
            # =========================================================================
            filled_mask = shiba_mask.copy()

            # 尋找最外層輪廓
            contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 將所有外部輪廓包圍的區域「一次塗滿白色」
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
            # =========================================================================
            # 後續修飾：移除邊緣噪點
            # =========================================================================
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            # 先做開運算斷開與邊框的黏連
            opened_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
            # 再做微侵蝕讓輪廓更漂亮
            eroded_mask = cv2.erode(opened_mask, kernel_ellipse, iterations=1)

            # 步驟三：使用 bitwise_and 與遮罩合併，提取出「只有柴犬貼紙」且「其餘背景全黑」的超乾淨影像
            target_512_cleaned = cv2.bitwise_and(target_512, target_512, mask=eroded_mask)

            # =========================================================================
            # 將淨化後的 target_512_cleaned 送入你原本的影像增強與二值化流程
            # =========================================================================
            gray = cv2.cvtColor(target_512_cleaned, cv2.COLOR_BGR2GRAY)
            #gamma1 = 2.2
            # 影像增強
            lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
            gray_gamma = cv2.LUT(gray, lut)
            target_gamma_CLAHE = _CLAHE.apply(gray_gamma)
            blur1 = cv2.GaussianBlur(target_gamma_CLAHE, (5, 5), 1.0)
            mask = cv2.subtract(gray_gamma, blur1)
            k = 1.5
            sharpened = cv2.addWeighted(gray_gamma, 1.0, mask, k, 0)
            blur_final = cv2.GaussianBlur(sharpened, (5, 5), 0)

            # 💡 關鍵修正：原本是 THRESH_BINARY_INV，現在因為背景全黑，必須改成 THRESH_BINARY！
            # 這樣才能確保「黑背景維持黑色(0)」，「柴犬貼紙變成白色(255)」
            _, otsu = cv2.threshold(blur_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # MORPH_CLOSE (閉運算)：填滿柴犬貼紙內部細微縫隙
            cleaned_binary = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, _KERNEL, iterations=1)
            # MORPH_OPEN (開運算)：去除殘留的微小點雜訊
            cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, _KERNEL, iterations=1)

            edges= auto_canny(cleaned_binary)
            final_dilate = cv2.dilate(edges, _KERNEL, iterations=1)

            # 4. 獲取輪廓 (複用現有過濾邏輯)
            hsv_contours = get_contours(final_dilate, noise_floor)
            # =========================================================================
            # 核心優化 3：自適應矩形框選與面積 PutText 打印
            # =========================================================================
            output_hsv = source_BGR.copy()
            all_contours_img = source_BGR.copy()

            # =========================================================================
            # 配置整張影像左上角數據清單的起始位置與間距（自適應大圖解析度）
            # =========================================================================
            list_x = max(20, int(25 * ratio))       # 距離左邊界的間隔
            list_y = max(30, int(35 * ratio))       # 第一筆資料的起始高度
            line_spacing = max(20, int(25 * ratio)) # 每行資料的跳行間距

            for idx, c in enumerate(hsv_contours):
                # 計算當前目標在 512 尺度下的精準面積
                area = cv2.contourArea(c)

                # 將座標等比例還原至原始大圖影像大小
                rescaled_c = (c * ratio).astype(np.int32)

                # 計算自適應線條與文字粗細
                thickness = max(2, int(2 * ratio))
                font_thickness = max(1, int(1.5 * ratio))
                font_scale = 0.4 * ratio

                # -------------------------------------------------------------------------
                # 1. 繪製目標物件「真實被包圍的不規則輪廓」
                # -------------------------------------------------------------------------
                cv2.drawContours(all_contours_img, [rescaled_c], -1, (0, 0, 255), thickness)

                # -------------------------------------------------------------------------
                # 2. 計算不規則輪廓的幾何中心點 (Centroid)
                # -------------------------------------------------------------------------
                M = cv2.moments(rescaled_c)
                if M["m00"] != 0:
                    # 透過數學矩計算真正的質心
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    # 邊緣極端狀況下若面積為0，則退回使用邊界矩形中心
                    x, y, w, h = cv2.boundingRect(rescaled_c)
                    center_x = x + w // 2
                    center_y = y + h // 2

                # -------------------------------------------------------------------------
                # 3. 在目標輪廓的中心點標註純編號 (例如: "1", "2")
                # 💡 這裡將字體稍微放大 (font_scale * 1.2) 讓編號在中心更清晰，並做微小偏移置中
                # -------------------------------------------------------------------------
                id_text = f"{idx+1}"
                offset_x = int(5 * ratio)
                offset_y = int(5 * ratio)
                cv2.putText(all_contours_img, id_text, (center_x - offset_x, center_y + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # -------------------------------------------------------------------------
                # 4. 依照順序在整張影像的左上角打印詳細數據清單
                # -------------------------------------------------------------------------
                data_text = f"{idx+1}:{int(area)}px"
                cv2.putText(all_contours_img, data_text, (list_x, list_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # 【關鍵跳行機制】每打印完一筆，將下一次的 Y 座標往下推一個間距
                list_y += line_spacing

            # 將最終繪製成果進行編碼送回 App 介面
            _, buf = cv2.imencode(".png", all_contours_img)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "LAB_findContour":
            filtered_img = cv2.bilateralFilter(target_512, d=9, sigmaColor=75, sigmaSpace=75)
            # ================= 新增：提取 V 通道做 CLAHE，再拼回 HSV =================
            # 步驟 A：轉到 LAB 色彩空間優化「亮度(L)」，完美保留色彩比例不偏色
            lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            # -------------------------------------------------------------------------
            # 【新增功能 2】: LAB 空間的動態 Gamma 矯正 (專門應對過曝/欠曝環境)
            # gamma_val_lab 可由 App UI 介面動態傳入。
            # Convention: > 1.0 增亮環境, < 1.0 壓暗環境
            # gamma_val = 2.0
            # -------------------------------------------------------------------------
            if gamma_val != 1.0:
                inv_gamma = 1.0 / gamma_val
                lut_lab = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
                l_ch = cv2.LUT(l_ch, lut_lab)  # 直接對亮度通道做 Gamma 矯正

            l_enhanced = _CLAHE.apply(l_ch)  # 對亮度通道進行自適應直方圖等化
            # 重新拼回優化後的 LAB
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])

            # 直接對 LAB 影像進行範圍篩選
            # L: 40~255 (過濾黑手機), A: 130~255 (微偏紅), B: 145~255 (強烈偏黃)
            lower_shiba_lab = np.array([40, 130, 145])
            upper_shiba_lab = np.array([255, 255, 255])
            # 產生遮罩 (這就是新的 skin_mask)
            skin_mask = cv2.inRange(lab_enhanced, lower_shiba_lab, upper_shiba_lab)
            # 2. 【新增：白色遮罩】抓取貼紙內外的純白與淺色區域
            # 原理：L 亮度極高，A 和 B 緊咬在 128 附近 (容許度設在 115 ~ 140)
            lower_white_lab = np.array([160, 115, 115])
            upper_white_lab = np.array([255, 142, 142])
            white_mask = cv2.inRange(lab_enhanced, lower_white_lab, upper_white_lab)
            # =========================================================================
            # 核心解法：利用「黃褐色毛框」動態生成安全區，強制切斷 MagSafe 環
            # =========================================================================

            # 【步驟 A】將精準的柴犬毛色向外均勻擴張一圈，定義出「柴犬貼紙白邊可能存在的合理區域」
            # 這裡的核心尺寸 (21, 21) 決定了允許白邊向外延伸的極限，可依貼紙白邊寬度調整
            kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            shiba_zone = cv2.dilate(skin_mask, kernel_zone, iterations=2)
            # 【步驟 B】關鍵交集過濾！只有落在「柴犬安全區域內」的白色才被允許保留
            # 這樣一來，懸空在背景、沒有挨著柴犬毛色的 90% MagSafe 磁吸環會被瞬間抹殺、蒸發！
            clean_white_mask = cv2.bitwise_and(white_mask, shiba_zone)

            # 【步驟 C】將純毛色與洗乾淨的白邊進行聯集，這才是完美的 combined_mask
            combined_mask = cv2.bitwise_or(skin_mask, clean_white_mask)
            #---------------------------------------------------------------------------------
            # 這個步驟能強行把相機動態畫面中，所有因為反光、噪點而產生的微小裂縫「黏合」起來
            kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            healed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_bridge)

            # 使用黏合完整、無斷口的 healed_mask 來做複製與輪廓尋找
            filled_mask = healed_mask.copy()

            # # 步驟 2：尋找所有目標物件的「最外層輪廓」(cv2.RETR_EXTERNAL)
            # # 這步會自動無視柴犬身體內部的黑色空洞與線條，只鎖定最外圍的邊界
            contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 步驟 3：將最外層輪廓的內部「全部填滿白色」 (thickness = -1)
            # 這能以最直覺的方式，把貼紙內部瞬間變成厚實的實心白區塊，且完全不傷及外圈細節！
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

            # 步驟 4：精細修邊（可選）
            # 因為內部已經是 100% 實心，如果你想消除邊緣極微小的毛刺或獨立小雜點，
            # 只要做一次非常微小核心的開運算或侵蝕即可，貼紙本體絕對不會再被破壞。
            kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_fine)
            # 步驟三：使用 bitwise_and 與遮罩合併，提取出「只有柴犬貼紙」且「其餘背景全黑」的超乾淨影像
            target_512_cleaned = cv2.bitwise_and(target_512, target_512, mask=eroded_mask)

            # =========================================================================
            # 將淨化後的 target_512_cleaned 送入你原本的影像增強與二值化流程
            # =========================================================================
            gray = cv2.cvtColor(target_512_cleaned, cv2.COLOR_BGR2GRAY)
            #gamma1 = 2.2
            # 影像增強
            lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
            gray_gamma = cv2.LUT(gray, lut)
            target_gamma_CLAHE = _CLAHE.apply(gray_gamma)
            blur1 = cv2.GaussianBlur(target_gamma_CLAHE, (5, 5), 1.0)
            mask = cv2.subtract(gray_gamma, blur1)
            k = 1.5
            sharpened = cv2.addWeighted(gray_gamma, 1.0, mask, k, 0)
            blur_final = cv2.GaussianBlur(sharpened, (5, 5), 0)

            # 💡 關鍵修正：原本是 THRESH_BINARY_INV，現在因為背景全黑，必須改成 THRESH_BINARY！
            # 這樣才能確保「黑背景維持黑色(0)」，「柴犬貼紙變成白色(255)」
            _, otsu = cv2.threshold(blur_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # -------------------------------------------------------------------------
            # 【核心修正 1】: 顛倒形態學順序，阻斷外部 MagSafe 殘留雜訊黏邊
            # -------------------------------------------------------------------------
            # 優先步驟一：先進行開運算 (MORPH_OPEN)，徹底瓦解、抹殺外圍孤立的點狀與環狀雜訊
            cleaned_binary = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, _KERNEL, iterations=1)
            # 優先步驟二：雜訊消失後，再做閉運算 (MORPH_CLOSE)，這時只會修復貼紙內部線條，絕不黏連
            cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, _KERNEL, iterations=1)

            edges = auto_canny(cleaned_binary)
            final_dilate = cv2.dilate(edges, _KERNEL, iterations=1)

            # 4. 獲取輪廓 (複用現有過濾邏輯)
            hsv_contours = get_contours(final_dilate, noise_floor)

            # =========================================================================
            # 核心優化 3：自適應矩形框選與面積 PutText 打印
            # =========================================================================
            output_lab = source_BGR.copy()
            all_contours_img = source_BGR.copy()

            # =========================================================================
            # 配置整張影像左上角數據清單的起始位置與間距（自適應大圖解析度）
            # =========================================================================
            list_x = max(20, int(25 * ratio))       # 距離左邊界的間隔
            list_y = max(30, int(35 * ratio))       # 第一筆資料的起始高度
            line_spacing = max(20, int(25 * ratio)) # 每行資料的跳行間距

            for idx, c in enumerate(hsv_contours):
                # 計算當前目標在 512 尺度下的精準面積
                area = cv2.contourArea(c)

                # 將座標等比例還原至原始大圖影像大小
                rescaled_c = (c * ratio).astype(np.int32)

                # 計算自適應線條與文字粗細
                thickness = max(2, int(2 * ratio))
                font_thickness = max(1, int(1.5 * ratio))
                font_scale = 0.4 * ratio

                # -------------------------------------------------------------------------
                # 1. 繪製目標物件「真實被包圍的不規則輪廓」
                # -------------------------------------------------------------------------
                cv2.drawContours(all_contours_img, [rescaled_c], -1, (0, 0, 255), thickness)

                # -------------------------------------------------------------------------
                # 2. 計算不規則輪廓的幾何中心點 (Centroid)
                # -------------------------------------------------------------------------
                M = cv2.moments(rescaled_c)
                if M["m00"] != 0:
                    # 透過數學矩計算真正的質心
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    # 邊緣極端狀況下若面積為0，則退回使用邊界矩形中心
                    x, y, w, h = cv2.boundingRect(rescaled_c)
                    center_x = x + w // 2
                    center_y = y + h // 2

                # -------------------------------------------------------------------------
                # 3. 在目標輪廓的中心點標註純編號 (例如: "1", "2")
                # 💡 這裡將字體稍微放大 (font_scale * 1.2) 讓編號在中心更清晰，並做微小偏移置中
                # -------------------------------------------------------------------------
                id_text = f"{idx+1}"
                offset_x = int(5 * ratio)
                offset_y = int(5 * ratio)
                cv2.putText(all_contours_img, id_text, (center_x - offset_x, center_y + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # -------------------------------------------------------------------------
                # 4. 依照順序在整張影像的左上角打印詳細數據清單
                # -------------------------------------------------------------------------
                data_text = f"{idx+1}:{int(area)}px"
                cv2.putText(all_contours_img, data_text, (list_x, list_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # 【關鍵跳行機制】每打印完一筆，將下一次的 Y 座標往下推一個間距
                list_y += line_spacing

            # 將最終繪製成果進行編碼送回 App 介面
            _, buf = cv2.imencode(".png", all_contours_img)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "LAB_Debug":
            # --- LAB 偵錯模式：輸出 LAB 流程中的 target_gamma_CLAHE ---
            filtered_img = cv2.bilateralFilter(target_512, d=9, sigmaColor=75, sigmaSpace=75)
            lab = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            
            if gamma_val != 1.0:
                inv_gamma = 1.0 / gamma_val
                lut_lab = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
                l_ch = cv2.LUT(l_ch, lut_lab)

            l_enhanced = _CLAHE.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])

            lower_shiba_lab = np.array([40, 130, 145])
            upper_shiba_lab = np.array([255, 255, 255])
            skin_mask = cv2.inRange(lab_enhanced, lower_shiba_lab, upper_shiba_lab)
            
            lower_white_lab = np.array([160, 115, 115])
            upper_white_lab = np.array([255, 142, 142])
            white_mask = cv2.inRange(lab_enhanced, lower_white_lab, upper_white_lab)
            
            kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            shiba_zone = cv2.dilate(skin_mask, kernel_zone, iterations=2)

            clean_white_mask = cv2.bitwise_and(white_mask, shiba_zone)
            combined_mask = cv2.bitwise_or(skin_mask, clean_white_mask)

            kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            healed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_bridge)

            # 使用黏合完整、無斷口的 healed_mask 來做複製與輪廓尋找
            filled_mask = healed_mask.copy()

            contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

            kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_fine)
            target_512_cleaned = cv2.bitwise_and(target_512, target_512, mask=eroded_mask)

            gray = cv2.cvtColor(target_512_cleaned, cv2.COLOR_BGR2GRAY)
            lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
            gray_gamma = cv2.LUT(gray, lut)
            target_gamma_CLAHE_lab = _CLAHE.apply(gray_gamma)

            debug_view = cv2.resize( target_512_cleaned , (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode(".png", debug_view)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "Contour Detection":
            all_contours_img = source_BGR.copy()
            for idx, c in enumerate(target_contours):
                area = cv2.contourArea(c)
                rescaled_c = (c * ratio).astype(np.int32)
                x, y, w, h = cv2.boundingRect(rescaled_c)
                
                # 只保留這一次繪製，使用還原後的座標 [rescaled_c]
                cv2.drawContours(all_contours_img, [rescaled_c], -1, (0, 0, 255), 2)

                text = f" {idx+1}: {area:.0f}px"
                text_y = y - 10 if y - 10 > 15 else y + 20
                #cv2.putText(all_contours_img, text, (x, text_y),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ratio, (0, 255, 255), int(1 * ratio), cv2.LINE_AA)
            _, buf = cv2.imencode(".png", all_contours_img)
            result_data["image"] = buf.tobytes()
            return result_data
        elif mode == "Object Recognition":
            if 'contour' not in template_cache:
                template_cache['contour'] = get_template_contour(TEMPLATE_PATH)
            template_contour = template_cache['contour']
            best_score, best_contour = 999.0, None
            for contour in target_contours:
                score = match_score(template_contour, contour)
                if score < best_score:
                    best_score, best_contour = score, contour
            output = source_BGR.copy()
            if best_contour is not None and best_score < MATCH_THRESHOLD:
                m = cv2.moments(best_contour)
                if m['m00'] != 0:
                    angle = get_angle(m)
                    result_data["angle"] = angle
                    rescaled_contour = (best_contour * ratio).astype(np.int32)
                    cx = int((m['m10'] / m['m00']) * ratio)
                    cy = int((m['m01'] / m['m00']) * ratio)
                    cv2.drawContours(output, [rescaled_contour], -1, (0, 255, 0), int(3 * ratio))
                    cv2.putText(output, f"OK: {best_score:.4f}", (cx - int(80*ratio), cy - int(40*ratio)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ratio, (0, 255, 0), int(1.5 * ratio), cv2.LINE_AA)
                    result_data["status"] = "OK"
            else:
                result_data["status"] = "NO_MATCH"
                output = cv2.cvtColor(cv2.resize(dilate, (w_orig, h_orig)), cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode(".png", output)
            result_data["image"] = buf.tobytes()
            return result_data
    except Exception as e:
        result_data["status"], result_data["error_msg"] = "ERROR", str(e)
    return result_data
