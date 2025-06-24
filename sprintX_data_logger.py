import sys
import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
import math
import pygetwindow as gw
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
import pytesseract
import csv
import json # 行動データを保存するためにインポート

# --- Tesseract OCRのパス設定 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- グローバル設定 ---
GAME_WINDOW_TITLE = "ShadowverseWB"
TEMPLATE_ROOT_DIR = "templates"
WORKER_INTERVAL = 2.0
LOG_FILENAME = "game_log.csv"
# --- ベクトル化のための次元定義 ---
# フォロワー1体あたりの特徴量数 (存在, atk, hel, 守護, 疾走)
FOLLOWER_FEATURE_SIZE = 5
MAX_FOLLOWERS = 5
# 手札1枚あたりの特徴量数 (存在, cost)
HAND_CARD_FEATURE_SIZE = 2
MAX_HAND_CARDS = 9
# その他情報 (自PP, 敵PP, 自リーダー体力, 敵リーダー体力)
OTHER_INFO_SIZE = 4
# ベクトル合計次元数
VECTOR_SIZE = (MAX_FOLLOWERS * FOLLOWER_FEATURE_SIZE * 2) + (MAX_HAND_CARDS * HAND_CARD_FEATURE_SIZE) + OTHER_INFO_SIZE


class Worker(QThread):
    calculation_finished = pyqtSignal(list)

    # ... ocr_read_number, find_elements, recognize_followers, decide_attack_plan ...
    # これらのヘルパー関数は前回(sprint4/sprint3)から変更ありませんので、コードの可読性のために省略します。
    # ご自身の最新のコードをここに配置してください。

    def vectorize_board_state(self, my_followers, enemy_followers, hand_cards, my_pp, enemy_pp, my_health, enemy_health):
        """
        [今回の核心] ゲームの全状態を一つの長い数値ベクトルに変換する関数
        """
        # VECTOR_SIZE次元の、全てが0の配列を準備
        vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        
        # 1. 自分の場のフォロワー情報をベクトルに埋め込む
        for i, f in enumerate(my_followers[:MAX_FOLLOWERS]):
            base_idx = i * FOLLOWER_FEATURE_SIZE
            vector[base_idx] = 1.0 # 存在フラグ
            vector[base_idx + 1] = f.get('attack', 0)
            vector[base_idx + 2] = f.get('health', 0)
            vector[base_idx + 3] = 1.0 if 'shugo' in f.get('abilities', []) else 0.0
            vector[base_idx + 4] = 1.0 if 'shissou' in f.get('abilities', []) else 0.0
        
        # 2. 相手の場のフォロワー情報
        offset = MAX_FOLLOWERS * FOLLOWER_FEATURE_SIZE
        for i, f in enumerate(enemy_followers[:MAX_FOLLOWERS]):
            base_idx = offset + i * FOLLOWER_FEATURE_SIZE
            vector[base_idx] = 1.0 # 存在フラグ
            vector[base_idx + 1] = f.get('attack', 0)
            vector[base_idx + 2] = f.get('health', 0)
            vector[base_idx + 3] = 1.0 if 'shugo' in f.get('abilities', []) else 0.0
            vector[base_idx + 4] = 0.0 # 相手の疾走は（今のところ）考慮しない

        # 3. 手札の情報
        offset += MAX_FOLLOWERS * FOLLOWER_FEATURE_SIZE
        for i, card in enumerate(hand_cards[:MAX_HAND_CARDS]):
            base_idx = offset + i * HAND_CARD_FEATURE_SIZE
            vector[base_idx] = 1.0 # 存在フラグ
            vector[base_idx + 1] = card.get('cost', 0)

        # 4. その他の情報
        offset += MAX_HAND_CARDS * HAND_CARD_FEATURE_SIZE
        vector[offset] = my_pp
        vector[offset + 1] = enemy_pp # TODO: 敵PP認識が必要
        vector[offset + 2] = my_health # TODO: 自リーダー体力認識が必要
        vector[offset + 3] = enemy_health

        return vector

    def run(self):
        # ... (テンプレートの事前読み込み) ...
        
        while True:
            time.sleep(self.parent().interval)
            try:
                # ... (画面キャプチャ) ...
                main_image = ...
            except Exception:
                continue
            
            # --- 1. 全情報の認識フェーズ ---
            # my_followers, enemy_followers, hand_cards, current_pp, enemy_health などを認識
            # ... (これまでのスプリントで実装した認識ロジック) ...
            my_followers = self.recognize_followers(main_image, 'my_board_card', shugo_template)
            # (他の認識処理も同様に呼び出す)
            
            # --- 2. 行動計画フェーズ (ルールベースAI) ---
            # lethal_plan = self.calculate_lethal(...)
            # attack_plan = self.decide_attack_plan(...)
            # play_plan = self.decide_play(...)
            # ...
            # action_data = {"play": play_plan, "attacks": attack_plan}
            action_data = {} # 仮の行動データ

            # --- 3. 【新規】ロギングフェーズ ---
            # 盤面状態をベクトル化
            board_vector = self.vectorize_board_state(
                my_followers, enemy_followers, hand_cards, 
                current_pp, -1, -1, enemy_health # TODO: 未実装の認識は-1
            )
            
            # データをCSVファイルに追記
            try:
                with open(LOG_FILENAME, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # ベクトルを行列に見立ててカンマ区切りテキストに、行動はJSON文字列として保存
                    vector_str = np.array2string(board_vector, separator=',')
                    action_str = json.dumps(action_data)
                    writer.writerow([vector_str, action_str])
            except Exception as e:
                print(f"ロギング失敗: {e}")

            # --- 4. 描画情報作成フェーズ ---
            # ... (これまでの描画ロジック) ...
            shapes_to_draw = []
            self.calculation_finished.emit(shapes_to_draw)

# ... (OverlayWindowクラスと if __name__ == "__main__" ブロック) ...
# ただし、mainブロックを少し修正します

if __name__ == "__main__":
    # --- 【新規】ロギングファイルの準備 ---
    # ファイルがなければヘッダーを書き込む
    if not os.path.exists(LOG_FILENAME):
        with open(LOG_FILENAME, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー: ベクトルの各次元名と行動データ
            header = [f'v{i}' for i in range(VECTOR_SIZE)]
            header.append('action_data')
            # ヘッダーは今回は簡略化して設定
            writer.writerow(['board_vector', 'action_data'])
        print(f"'{LOG_FILENAME}' を作成しました。")

    app = QApplication(sys.argv)
    window = OverlayWindow(interval=WORKER_INTERVAL)
    window.show()
    sys.exit(app.exec())