import cv2
import numpy as np
from PIL import ImageGrab
import os
import pygetwindow as gw
import pytesseract # OCRライブラリをインポート

# --- ステップ1：Tesseract本体へのパスを設定 ---
# ご自身の環境に合わせてパスを修正してください
# Windowsの例:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# macOS (Apple Silicon)の例:
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


# --- ステップ2：OCRヘルパー関数を定義 ---
def ocr_read_number(image: np.ndarray) -> int:
    """
    画像から数字を1つ読み取るためのヘルパー関数。
    OCRの精度を上げるための前処理もここで行う。
    """
    try:
        # 1. グレースケール化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. リサイズ（小さすぎる画像はOCRが苦手なため、少し拡大する）
        height, width = gray.shape
        scale_factor = 3
        resized_gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)

        # 3. 二値化（白と黒の2色に変換）
        #    THRESH_BINARY_INV: 色を反転（数字が白、背景が黒になるように）
        #    THRESH_OTSU: 最適な閾値を自動で計算してくれる
        _, thresh = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # (デバッグ用) 前処理後の画像を保存して確認したい場合は以下のコメントを外す
        # cv2.imwrite("debug_ocr_preprocess.png", thresh)

        # 4. TesseractでOCR実行
        #   --psm 7: 画像を1行として扱うモード
        #   tessedit_char_whitelist=0123456789: 読み取り対象を数字に限定
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(thresh, config=config, lang='eng') # lang='eng'も指定すると良い
        
        return int(text.strip())
    except (ValueError, TypeError):
        # 読み取れない、または数字に変換できない場合
        return -1


# find_elements関数はsprint1_advanced.pyからそのままコピー
def find_elements(main_image, template_image, threshold=0.8):
    # ... (この関数の中身は変更なし) ...
    h, w = template_image.shape[:2]
    result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    found_points = []
    for pt in zip(*locations[::-1]):
        found_points.append((pt[0], pt[1], w, h))
    return found_points


if __name__ == "__main__":
    GAME_WINDOW_TITLE = "ShadowverseWB"
    # PPを検出するためのテンプレートカテゴリ
    PP_CATEGORY = 'my_pp' 
    TEMPLATE_ROOT_DIR = "templates"

    # --- ゲームウィンドウのキャプチャ ---
    try:
        game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
        x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        print("ゲーム画面のキャプチャに成功しました。")
    except Exception as e:
        print(f"エラー: ゲーム画面のキャプチャに失敗しました。: {e}")
        exit()

    # --- PPテンプレートのマッチング ---
    found_pp_areas = []
    pp_category_path = os.path.join(TEMPLATE_ROOT_DIR, PP_CATEGORY)
    if os.path.isdir(pp_category_path):
        for template_filename in os.listdir(pp_category_path):
            template_path = os.path.join(pp_category_path, template_filename)
            template_image = cv2.imread(template_path)
            if template_image is not None:
                # 閾値を少し高め(0.9)に設定して、誤検出を減らす
                found_pp_areas.extend(find_elements(main_image, template_image, threshold=0.7))

    # --- ステップ3：PPの領域を切り出し、OCRを実行 ---
    if not found_pp_areas:
        print("PPの表示領域が見つかりませんでした。テンプレート画像を確認してください。")
    else:
        # 複数見つかった場合でも、最初のものだけを対象とする
        x, y, w, h = found_pp_areas[0]
        print(f"PPの領域を座標 (x={x}, y={y}) に発見しました。")

        # 1. PP領域を切り出す
        pp_image = main_image[y:y+h, x:x+w]

        # 2.【重要】デバッグ用に切り出した画像を保存する
        cv2.imwrite("debug_pp_image.png", pp_image)
        print("OCRにかける画像を 'debug_pp_image.png'として保存しました。")

        # 3. OCR関数を呼び出して数値を読み取る
        read_pp_value = ocr_read_number(pp_image)

        # 4. 結果を表示する
        if read_pp_value != -1:
            print("------------------------------------")
            print(f"🎉 読み取り成功！ 現在のPP: {read_pp_value}")
            print("------------------------------------")
        else:
            print("------------------------------------")
            print("❗️ 読み取り失敗...。'debug_pp_image.png' を確認してください。")
            print("考えられる原因：画像が不鮮明、切り出し範囲が不正確、など。")
            print("------------------------------------")

        # 結果を画面上で確認するために、見つけた場所に四角を描画
        cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("Detection Result", main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()