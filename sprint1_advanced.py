import cv2
import numpy as np
from PIL import ImageGrab
import os
import sys
import io
import pygetwindow as gw # ウィンドウ操作ライブラリ

# 出力のエンコーディングをUTF-8に強制
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 設定項目 ---
# テンプレート画像が保存されているルートフォルダ
TEMPLATE_ROOT_DIR = "templates"
# 結果を保存するフォルダ
OUTPUT_DIR = "outputs"
# ゲームウィンドウのタイトル
GAME_WINDOW_TITLE = "ShadowverseWB" # 対象のゲームに合わせて変更

def find_elements(main_image, template_image, threshold=0.7):
    """ 複数箇所に一致するテンプレートを探す（重複を避けるための改良は今後の課題） """
    h, w = template_image.shape[:2]
    result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    
    found_points = []
    for pt in zip(*locations[::-1]):
        found_points.append((pt[0], pt[1], w, h))
        
    # TODO: 重複する領域をまとめる処理（Non-Maximum Suppressionなど）を後で追加するとより良い
    return found_points

if __name__ == "__main__":
    # --- 1. ゲームウィンドウの検索とキャプチャ ---
    try:
        print(f"'{GAME_WINDOW_TITLE}'のウィンドウを探しています...")
        game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
    except IndexError:
        print(f"エラー: '{GAME_WINDOW_TITLE}'のウィンドウが見つかりません。ゲームを起動してください。")
        exit()

    print("ウィンドウを発見。スクリーンショットを撮影します。")
    # ウィンドウの座標を取得
    x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
    # Pillowを使って指定領域のみキャプチャ
    screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
    main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # --- 2. 構造化されたデータとして要素を検索 ---
    # 結果を保存するための辞書
    found_elements = {}

    print("テンプレートのマッチングを開始します...")
    # templatesフォルダ内の各カテゴリフォルダをループ
    for category in os.listdir(TEMPLATE_ROOT_DIR):
        category_path = os.path.join(TEMPLATE_ROOT_DIR, category)
        if not os.path.isdir(category_path):
            continue
        
        found_elements[category] = []
        
        # カテゴリ内の各テンプレート画像をループ
        for template_filename in os.listdir(category_path):
            if not template_filename.endswith(".png"):
                continue
            
            template_path = os.path.join(category_path, template_filename)
            
            # 日本語などのマルチバイト文字を含むファイルパスに対応するため、np.fromfileとcv2.imdecodeを使用
            n = np.fromfile(template_path, np.uint8)
            template_image = cv2.imdecode(n, cv2.IMREAD_COLOR)

            # 画像が正しく読み込めたかチェック
            if template_image is None:
                print(f"警告: テンプレート画像が読み込めませんでした: {template_path}")
                continue
            
            # テンプレートを検索
            locations = find_elements(main_image, template_image, threshold=0.8)
            if locations:
                found_elements[category].extend(locations)

    print("マッチング完了。")
    print("\n--- 検出結果 ---")
    for category, items in found_elements.items():
        print(f"カテゴリ '{category}': {len(items)} 個")
    print("-----------------\n")

    # --- 3. 見つけた要素を切り出して保存 ---
    print("検出した要素を切り出して保存します...")
    # 出力ルートフォルダがなければ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    original_with_rects = main_image.copy()

    for category, items in found_elements.items():
        # カテゴリごとの出力フォルダを作成
        category_output_dir = os.path.join(OUTPUT_DIR, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
            
        # カテゴリごとに色を変えて四角を描画
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        for i, (x, y, w, h) in enumerate(items):
            # 四角を描画
            cv2.rectangle(original_with_rects, (x, y), (x + w, y + h), color, 2)
            
            # 画像を切り出す (Numpyスライシング)
            cropped_image = main_image[y:y+h, x:x+w]
            
            # 切り出した画像を保存
            cropped_filename = f"{category}_{i}.png"
            cv2.imwrite(os.path.join(category_output_dir, cropped_filename), cropped_image)

    # 全ての検出結果を描画した画像を保存
    result_path = os.path.join(OUTPUT_DIR, "result_all.png")
    cv2.imwrite(result_path, original_with_rects)
    print(f"結果を '{OUTPUT_DIR}' フォルダに保存しました。")
    
    # 結果を表示
    cv2.imshow("Detection Result", original_with_rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()