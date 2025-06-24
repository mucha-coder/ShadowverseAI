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
import torch
import torch.nn as nn

# --- 前回と同じ設定 ---
GAME_WINDOW_TITLE = "ShadowverseWB"; TEMPLATE_ROOT_DIR = "templates"; WORKER_INTERVAL = 2.0
MODEL_SAVE_PATH = "imitation_model.pth"
VECTOR_SIZE = 81 # train_model.pyと合わせる
NUM_ACTIONS = 10 # train_model.pyと合わせる

# --- ステップ1：訓練時と同じモデル構造を定義 ---
class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        return self.network(x)

class Worker(QThread):
    calculation_finished = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        # --- ステップ2：学習済みモデルの読み込み ---
        print("AIの脳（学習済みモデル）を読み込んでいます...")
        self.model = ImitationModel(VECTOR_SIZE, NUM_ACTIONS)
        try:
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            self.model.eval() # ★★★ 推論モードに切り替え ★★★
            print("モデルの読み込みに成功しました。")
        except FileNotFoundError:
            print(f"エラー: 学習済みモデル '{MODEL_SAVE_PATH}' が見つかりません。")
            print("先に `train_model.py` を実行して、モデルを訓練してください。")
            self.model = None

    # ... (認識ヘルパー関数群: ocr_read_number, find_elements, vectorize_board_state などはここに配置) ...
    
    def run(self):
        if self.model is None:
            return # モデルがなければ何もしない

        while True:
            time.sleep(self.parent().interval)
            try:
                # ... (画面キャプチャ) ...
                main_image = ...
            except Exception:
                continue

            # --- 1. 盤面認識とベクトル化 ---
            # ... (これまでのスプリントで実装した認識ロジック) ...
            # my_followers, enemy_followers, hand_cards, ... などを認識
            board_vector = self.vectorize_board_state(...)

            # --- 2. 【核心】機械学習モデルによる推論 ---
            with torch.no_grad(): # 勾配計算をオフにし、推論を高速化
                # ベクトルをPyTorchのテンソルに変換
                input_tensor = torch.from_numpy(board_vector).float().unsqueeze(0) # バッチ次元を追加
                
                # モデルに盤面情報を入力し、行動の評価値を取得
                output = self.model(input_tensor)
                
                # 最も評価値の高い行動を選択
                predicted_action_id = torch.argmax(output, dim=1).item()

            # --- 3. 行動の解釈と描画情報の作成 ---
            # TODO: predicted_action_id（例: 5）を、具体的なゲーム行動
            # （例: 「手札の6枚目をハイライト」）に変換するロジックをここに実装
            shapes_to_draw = self.interpret_action(predicted_action_id, hand_cards, my_followers)
            
            self.calculation_finished.emit(shapes_to_draw)

    def interpret_action(self, action_id, hand_cards, my_followers):
        """ [新規] 行動IDを具体的な描画命令に変換する関数 """
        shapes = []
        # 例：行動ID 0-8 は手札のプレイに対応
        if 0 <= action_id < 9 and action_id < len(hand_cards):
            card_to_play = hand_cards[action_id]
            shapes.append({"type":"rectangle", "rect":card_to_play['rect'], "color":(0,255,0,220)})
        # 他の行動ID（攻撃など）の解釈もここに追加
        
        return shapes

# ... (OverlayWindowクラスと if __name__ == "__main__" ブロックは変更なし) ...