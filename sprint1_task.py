# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import ImageGrab
import os
import sys
import io
import pygetwindow as gw

# 出力のエンコーディングをUTF-8に強制
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 設定項目 ---
# テンプレート画像が保存されているフォルダ
TEMPLATE_DIR = "templates"
# 結果を保存するフォルダ
OUTPUT_DIR = "outputs"

def find_element(main_image, template_image, threshold=0.7):
    """
    メイン画像からテンプレート画像と一致する箇所を探す関数

    Args:
        main_image (numpy.ndarray): 探索対象のメイン画像
        template_image (numpy.ndarray): 探したいテンプレート画像
        threshold (float): 一致度の閾値（0.0〜1.0）

    Returns:
        list: 見つかった全ての領域の座標 [(x, y, w, h), ...]
    """
    # テンプレート画像の高さを取得
    h, w = template_image.shape[:2]

    # テンプレートマッチングを実行
    # TM_CCOEFF_NORMEDは0.0から1.0の正規化された相関係数を返す（1.0が完全一致）
    result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)

    # 閾値以上の場所を全て見つける
    locations = np.where(result >= threshold)
    
    # 見つかった場所の座標を整形する
    # `zip(*locations[::-1])` は (y, x) のリストを (x, y) に変換するテクニック
    found_points = []
    for pt in zip(*locations[::-1]):
        # ptは(x, y)座標
        found_points.append((pt[0], pt[1], w, h))
        
    return found_points


if __name__ == "__main__":
    # --- 1. 準備 ---
    # 結果保存用フォルダがなければ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # テンプレート画像フォルダがなければエラー
    if not os.path.exists(TEMPLATE_DIR):
        print(f"エラー: '{TEMPLATE_DIR}' フォルダが見つかりません。テンプレート画像を保存してください。")
        exit()

    # --- 2. シャドウバースウィンドウのスクリーンショット撮影 ---
    print("シャドウバースのウィンドウを検索しています...")
    # シャドウバースのウィンドウタイトルを部分一致で検索
    target_windows = [w for w in gw.getAllWindows() if "ShadowverseWB" in w.title]
    if not target_windows:
        print("エラー: 'Shadowverse' ウィンドウが見つかりません。ゲームを起動してください。")
        exit()
    # 最初に見つかったウィンドウを対象とする
    window = target_windows[0]
    print(f"ウィンドウ '{window.title}' を対象にスクリーンショットを撮影します...")

    # ウィンドウが最小化されている場合は復元
    if window.isMinimized:
        window.restore()

    # ウィンドウを前面に移動
    window.activate()

    # ウィンドウの位置とサイズを取得
    left, top, right, bottom = window.left, window.top, window.right, window.bottom

    # Pillowを使ってウィンドウ領域のスクリーンショットを取得
    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    # OpenCVで扱えるようにNumpy配列に変換 (RGB -> BGR)
    main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    print("撮影完了。")

    # --- 3. テンプレートのマッチングと結果の描画 ---
    # テンプレートフォルダ内の全ての.pngファイルを処理
    for template_filename in os.listdir(TEMPLATE_DIR):
        if not template_filename.endswith(".png"):
            continue

        template_path = os.path.join(TEMPLATE_DIR, template_filename)
        print(f"テンプレート '{template_filename}' を検索中...")
        
        # テンプレート画像を読み込み
        template_image = cv2.imread(template_path)
        if template_image is None:
            print(f"警告: '{template_path}' が読み込めませんでした。")
            continue

        # テンプレートを検索
        found_locations = find_element(main_image, template_image, threshold=0.85)

        if not found_locations:
            print(f" -> '{template_filename}' は見つかりませんでした。")
            continue
            
        print(f" -> {len(found_locations)} 個見つかりました。")

        # 見つかった全ての場所に四角形を描画
        for (x, y, w, h) in found_locations:
            # 赤い四角で囲む (色はBGR形式)
            cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # --- 4. 結果の表示と保存 ---
    # 結果の画像を保存
    output_path = os.path.join(OUTPUT_DIR, "result.png")
    cv2.imwrite(output_path, main_image)
    print(f"結果を '{output_path}' に保存しました。")
    
    # 結果をウィンドウで表示（'q'キーで閉じる）
    cv2.imshow("Result", main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()