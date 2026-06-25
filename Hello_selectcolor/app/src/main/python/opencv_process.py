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

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = _keep(contours)
    
    if not result:
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

def canny_from_image_bytes(image_data: bytes, mode: str = "Contour Detection", gamma_val: float = 0.7, noise_floor: int = 300, target_idx: int = 0, hsv_ranges: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    主要處理入口。
    target_idx: 選定的目標輪廓編號 (1-N)，若為 0 則顯示所有。
    """
    result_data = {"image": None, "angle": 0.0, "status": "OK", "error_msg": ""}
    try:
        raw_buffer = np.frombuffer(image_data, dtype=np.uint8)
        source_BGR = cv2.imdecode(raw_buffer, cv2.IMREAD_COLOR)
        if source_BGR is None: raise ValueError("Decode failed")
        
        h_orig, w_orig = source_BGR.shape[:2]
        target_512 = cv2.resize(source_BGR, (BASE_SIZE, BASE_SIZE), interpolation=cv2.INTER_AREA)
        ratio = h_orig / float(BASE_SIZE)

        # --- 基礎預處理 (適用於所有模式) ---
        gray = cv2.cvtColor(target_512, cv2.COLOR_BGR2GRAY)
        
        # 影像增強流水線
        lut = np.array([((i / 255.0) ** (1.0/gamma_val if gamma_val != 0 else 1.0)) * 255 for i in range(256)], dtype=np.uint8)
        gray_gamma = cv2.LUT(gray, lut)
        target_gamma_CLAHE = _CLAHE.apply(gray_gamma)
        blur1 = cv2.GaussianBlur(target_gamma_CLAHE, (5, 5), 1.0)
        mask_sharp = cv2.subtract(gray_gamma, blur1)
        k_sharp = 1.5
        sharpened = cv2.addWeighted(gray_gamma, 1.0, mask_sharp, k_sharp, 0)
        blur_final = cv2.GaussianBlur(sharpened, (5, 5), 0)
        _, otsu = cv2.threshold(blur_final, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形態學優化
        cleaned_binary = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
        cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, _KERNEL, iterations=1)
        edges = auto_canny(cleaned_binary)
        dilate = cv2.dilate(edges, _KERNEL, iterations=1)
        
        # 獲取基礎候選輪廓
        target_contours = get_contours(dilate, noise_floor)

        # --- 預處理流程 (HSV 邏輯核心) ---
        def get_hsv_refined_contours(img_512, g_val, hsv_ranges=None):
            filtered = cv2.bilateralFilter(img_512, d=9, sigmaColor=75, sigmaSpace=75)
            hsv_img = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            
            if g_val != 1.0:
                invGamma = 1.0 / g_val
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                v = cv2.LUT(v, table)
            
            v_clahe = _CLAHE.apply(v)
            hsv_refined = cv2.merge([h, s, v_clahe])

            # 修正：處理來自 Java 的 ArrayList 類型，避免 len() 報錯
            is_empty = True
            try:
                if hsv_ranges is not None:
                    # 使用迭代檢查是否有內容，避開 len()
                    for _ in hsv_ranges:
                        is_empty = False
                        break
            except:
                is_empty = True

            if is_empty:
                # 預設：柴犬色 + 白色
                hsv_ranges_final = [
                    ([0, 32, 43], [25, 255, 255]),
                    ([0, 18, 150], [180, 55, 255])
                ]
            else:
                hsv_ranges_final = hsv_ranges

            mask_combined = None
            for range_item in hsv_ranges_final:
                # 確保 lower/upper 是 numpy array
                lower = np.array(range_item[0], dtype=np.uint8)
                upper = np.array(range_item[1], dtype=np.uint8)
                mask = cv2.inRange(hsv_refined, lower, upper)
                if mask_combined is None:
                    mask_combined = mask
                else:
                    mask_combined = cv2.bitwise_or(mask_combined, mask)

            k_guide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_eroded = cv2.erode(mask_combined, k_guide, iterations=1)

            # 輪廓填滿
            mask_filled = np.zeros_like(mask_eroded)
            cnts, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > 400:
                    cv2.drawContours(mask_filled, [c], -1, 255, thickness=cv2.FILLED)

            mask_opened_pre = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, k_guide, iterations=1)
            
            # --- 新增：針對提取後的影像再次使用 Otsu 處理燈光不均問題 ---
            # 1. 提取淨化影像 (僅保留遮罩內像素，背景全黑)
            target_cleaned = cv2.bitwise_and(img_512, img_512, mask=mask_opened_pre)
            
            # 2. 轉為灰階並套用影像增強
            gray_cleaned = cv2.cvtColor(target_cleaned, cv2.COLOR_BGR2GRAY)
            # 套用與全域一致的 Gamma 與 CLAHE
            lut_c = np.array([((i / 255.0) ** (1.0/g_val if g_val != 0 else 1.0)) * 255 for i in range(256)], dtype=np.uint8)
            gray_gamma_c = cv2.LUT(gray_cleaned, lut_c)
            gray_clahe_c = _CLAHE.apply(gray_gamma_c)
            
            # 3. 高斯模糊後套用 Otsu (由於背景已黑，改用 THRESH_BINARY)
            blur_final_c = cv2.GaussianBlur(gray_clahe_c, (5, 5), 0)
            _, otsu_refined = cv2.threshold(blur_final_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 4. 【核心修正 1】: 顛倒形態學順序 (閉運算 -> 開運算)
            # MORPH_CLOSE (閉運算)：填滿內部細微縫隙
            cleaned_binary = cv2.morphologyEx(otsu_refined, cv2.MORPH_CLOSE, _KERNEL, iterations=1)
            # MORPH_OPEN (開運算)：去除殘留的微小點雜訊
            cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_OPEN, _KERNEL, iterations=1)

            # 最終邊緣提取
            edges_final = auto_canny(cleaned_binary)
            dilate_final = cv2.dilate(edges_final, _KERNEL, iterations=1)
            return get_contours(dilate_final, noise_floor), cleaned_binary, dilate_final

        if mode == "Debug Pre-processing":
            # 獲取三筆資料，這裡我們觀察最原始的 Otsu 結果 (otsu_refined)
            _, _, dilate_final = get_hsv_refined_contours(target_512, gamma_val, hsv_ranges)
            
            # 將二值化影像還原至原始大小以供觀察
            debug_view = cv2.resize(dilate_final, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            _, buf = cv2.imencode(".png", debug_view)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "Contour Detection":
            all_contours_img = source_BGR.copy()
            for idx, c in enumerate(target_contours):
                rescaled_c = (c * ratio).astype(np.int32)
                cv2.drawContours(all_contours_img, [rescaled_c], -1, (0, 0, 255), 2)
            
            _, buf = cv2.imencode(".png", all_contours_img)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "HSV_findContour":
            hsv_cnts, _, _ = get_hsv_refined_contours(target_512, gamma_val, hsv_ranges)
            all_contours_img = source_BGR.copy()

            list_x = max(20, int(25 * ratio))
            list_y = max(30, int(35 * ratio))
            line_spacing = max(20, int(25 * ratio))

            for idx, c in enumerate(hsv_cnts):
                area = cv2.contourArea(c)
                rescaled_c = (c * ratio).astype(np.int32)
                thickness = max(2, int(2 * ratio))
                font_thickness = max(1, int(1.5 * ratio))
                font_scale = 0.4 * ratio

                cv2.drawContours(all_contours_img, [rescaled_c], -1, (0, 0, 255), thickness)

                M = cv2.moments(rescaled_c)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(rescaled_c)
                    center_x = x + w // 2
                    center_y = y + h // 2

                id_text = f"{idx+1}"
                offset_x = int(5 * ratio)
                offset_y = int(5 * ratio)
                cv2.putText(all_contours_img, id_text, (center_x - offset_x, center_y + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 255), font_thickness, cv2.LINE_AA)

                data_text = f"{idx+1}:{int(area)}px"
                cv2.putText(all_contours_img, data_text, (list_x, list_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

                list_y += line_spacing

            _, buf = cv2.imencode(".png", all_contours_img)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "Object Recognition":
            hsv_cnts, _, _ = get_hsv_refined_contours(target_512, gamma_val, hsv_ranges)
            
            if target_idx > 0 and target_idx <= len(hsv_cnts):
                template_c = hsv_cnts[target_idx - 1]
                best_s, best_i = 999.0, -1
                
                # 1. 執行匹配運算
                for i, c in enumerate(hsv_cnts):
                    score = match_score(template_c, c)
                    if score < best_s:
                        best_s, best_i = score, i
                
                if best_i != -1:
                    # 【核心構思】：匹配成功後，單獨分離出該輪廓圖案
                    # a. 建立單一物件的遮罩
                    single_mask = np.zeros((BASE_SIZE, BASE_SIZE), dtype=np.uint8)
                    cv2.drawContours(single_mask, [hsv_cnts[best_i]], -1, 255, thickness=cv2.FILLED)
                    
                    # b. 產生 target_cleaned (背景全黑，僅保留匹配成功的貼紙像素)
                    isolated_512 = cv2.bitwise_and(target_512, target_512, mask=single_mask)
                    
                    # c. 將影像還原至原始大小
                    output_matched = cv2.resize(isolated_512, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                    
                    # d. 在這張「乾淨」的影像上用綠色線繪製輪廓
                    matched_rescaled = (hsv_cnts[best_i] * ratio).astype(np.int32)
                    cv2.drawContours(output_matched, [matched_rescaled], -1, (0, 255, 0), 3)
                    
                    # e. 標註編號與面積 (關鍵修正：統一使用 512 尺度下的輪廓計算面積)
                    # 這樣數據才會與 HSV_findContour 模式完全一致
                    area_val = cv2.contourArea(hsv_cnts[best_i])
                    x, y, w, h = cv2.boundingRect(matched_rescaled)
                    
                    # 格式改為 (No.1 = 5000 px)，字體縮小 (0.5 * ratio)
                    label_text = f"(No.{best_i+1} = {int(area_val)} px)"
                    font_scale = 0.5 * ratio
                    
                    # 位置設定在輪廓以外的下方，並保留一些空隙 (y + h + 35*ratio)
                    text_x = x
                    text_y = y + h + int(35 * ratio)
                    
                    cv2.putText(output_matched, label_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, int(1.5 * ratio)), cv2.LINE_AA)
                    
                    _, buf = cv2.imencode(".png", output_matched)
                    result_data["image"] = buf.tobytes()
                    return result_data
            
            # --- 若無選定目標或匹配失敗，則顯示所有候選輪廓 (紅色) ---
            output_all = source_BGR.copy()
            for idx, c in enumerate(hsv_cnts):
                rescaled = (c * ratio).astype(np.int32)
                cv2.drawContours(output_all, [rescaled], -1, (0, 0, 255), 2)
                
                # 同步修改未選定狀態下的字體排版
                area_all = cv2.contourArea(rescaled)
                rx, ry, rw, rh = cv2.boundingRect(rescaled)
                label_all = f"(No.{idx+1} = {int(area_all)} px)"
                
                # 設定在下方並保留空隙
                cv2.putText(output_all, label_all, (rx, ry + rh + int(25 * ratio)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ratio, (0, 0, 255), 1, cv2.LINE_AA)

            _, buf = cv2.imencode(".png", output_all)
            result_data["image"] = buf.tobytes()
            return result_data

        if mode == "LAB_findContour":
            lab = cv2.cvtColor(target_512, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            if gamma_val != 1.0:
                inv_gamma = 1.0 / gamma_val
                lut_lab = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
                l_ch = cv2.LUT(l_ch, lut_lab)
            l_enhanced = _CLAHE.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            lower_lab = np.array([40, 130, 145])
            upper_lab = np.array([255, 255, 255])
            lab_mask = cv2.inRange(lab_enhanced, lower_lab, upper_lab)
            lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
            lab_cnts = get_contours(lab_mask, noise_floor)
            output = source_BGR.copy()
            for idx, c in enumerate(lab_cnts):
                rescaled = (c * ratio).astype(np.int32)
                x, y, w, h = cv2.boundingRect(rescaled)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output, f"#{idx+1}:{int(cv2.contourArea(c))}px", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ratio, (0, 0, 255), 1, cv2.LINE_AA)
            _, buf = cv2.imencode(".png", output)
            result_data["image"] = buf.tobytes()
            return result_data

        # --- 預設降級行為 ---
        all_contours_img = source_BGR.copy()
        for idx, c in enumerate(target_contours):
            rescaled = (c * ratio).astype(np.int32)
            cv2.drawContours(all_contours_img, [rescaled], -1, (0, 0, 255), 2)
        _, buf = cv2.imencode(".png", all_contours_img)
        result_data["image"] = buf.tobytes()
        return result_data
    except Exception as e:
        import traceback
        result_data["status"], result_data["error_msg"] = "ERROR", f"{str(e)}\n{traceback.format_exc()}"
    return result_data
