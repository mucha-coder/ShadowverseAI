import sys
import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
import pygetwindow as gw
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
import pytesseract

# --- Tesseract OCRのパス設定 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- グローバル設定 ---
TEMPLATE_ROOT_DIR = "templates"
GAME_WINDOW_TITLE = "ShadowverseWB"
WORKER_INTERVAL = 2.0 # 処理間隔

class Worker(QThread):
    calculation_finished = pyqtSignal(list)

    def ocr_read_number(self, image: np.ndarray) -> int:
        """ [移植した部品] 画像から数字を1つ読み取るためのヘルパー関数 """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scale_factor = 3
            resized_gray = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thresh, config=config, lang='eng')
            return int(text.strip())
        except (ValueError, TypeError):
            return -1

    def find_elements(self, main_image, template_image, threshold=0.8):
        h, w = template_image.shape[:2]
        result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        found_points = []
        for pt in zip(*locations[::-1]):
            found_points.append((pt[0], pt[1], w, h))
        return found_points

    def run(self):
        print("--- ワーカースレッド開始 ---")
        while True:
            time.sleep(self.parent().interval)
            main_image = None
            try:
                game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
                x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
                screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
                main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            except Exception:
                continue # ウィンドウが見つからなければ次のループへ

            # --- 1. PPの認識 ---
            current_pp = -1
            pp_template_path = os.path.join(TEMPLATE_ROOT_DIR, 'my_pp', 'template_pp_area.png') # PPテンプレートのパス
            if os.path.exists(pp_template_path):
                pp_template = cv2.imread(pp_template_path)
                pp_areas = self.find_elements(main_image, pp_template, threshold=0.9)
                if pp_areas:
                    x, y, w, h = pp_areas[0]
                    pp_image = main_image[y:y+h, x:x+w]
                    current_pp = self.ocr_read_number(pp_image)
            
            # デバッグ用プリント（もし動かなければコメントを外す）
            # print(f"現在のPP: {current_pp}")

            if current_pp == -1:
                self.calculation_finished.emit([]) # PPが読めなければ何も描画しない
                continue

            # --- 2. 手札とコストの認識 & 3. AIの判断 ---
            playable_cards = []
            cost_category_path = os.path.join(TEMPLATE_ROOT_DIR, 'cost')
            if os.path.isdir(cost_category_path):
                for template_filename in os.listdir(cost_category_path):
                    template_path = os.path.join(cost_category_path, template_filename)
                    cost_template = cv2.imread(template_path)
                    
                    found_costs_locations = self.find_elements(main_image, cost_template, threshold=0.9)
                    
                    for (cost_x, cost_y, cost_w, cost_h) in found_costs_locations:
                        # コスト部分の画像からコスト数値を読み取る
                        cost_image = main_image[cost_y:cost_y+cost_h, cost_x:cost_x+cost_w]
                        card_cost = self.ocr_read_number(cost_image)
                        
                        if card_cost != -1 and card_cost <= current_pp:
                            # カード全体の範囲を推定
                            OFFSET_X, OFFSET_Y = 15, 15
                            CARD_WIDTH, CARD_HEIGHT = 100, 100
                            card_rect = (cost_x - OFFSET_X, cost_y - OFFSET_Y, CARD_WIDTH, CARD_HEIGHT)
                            playable_cards.append({"rect": card_rect, "cost": card_cost})
            
            # デバッグ用プリント
            # print(f"プレイ可能なカード: {playable_cards}")

            # --- 4. 描画命令の生成 ---
            shapes_to_draw = []
            if playable_cards:
                # 最もコストが高いカードを選択
                best_card = max(playable_cards, key=lambda c: c["cost"])
                shapes_to_draw.append({
                    "type": "rectangle",
                    "rect": best_card["rect"],
                    "color": (0, 255, 0, 220) # 緑色でハイライト
                })
            
            # デバッグ用プリント
            # print(f"最終描画情報: {shapes_to_draw}")

            self.calculation_finished.emit(shapes_to_draw)

# OverlayWindowクラスとメインの実行部分はsprint2_prototype.pyから変更なし
class OverlayWindow(QMainWindow):
    def __init__(self, interval=2.0):
        super().__init__()
        self.interval = interval
        self.shapes_to_draw = []

        # ウィンドウの設定（最前面、フレームレス、透明、入力透過）
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool # タスクバーに表示しない
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # ゲームウィンドウに重ねる
        self.update_geometry()

        # ワーカースレッドの準備と開始
        self.worker = Worker()
        self.worker.setParent(self) # 親を設定
        self.worker.calculation_finished.connect(self.update_drawing)
        self.worker.start()

    def update_geometry(self):
        """ ゲームウィンドウの位置とサイズにUIを合わせる """
        try:
            game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
            self.setGeometry(game_window.left, game_window.top, game_window.width, game_window.height)
        except Exception as e:
            print(f"ウィンドウの位置調整に失敗: {e}")
            # ウィンドウが見つからない場合は非表示にする
            self.hide()

    def paintEvent(self, event):
        """ 描画イベント（ウィンドウの再描画が必要な時に呼ばれる） """
        painter = QPainter(self)
        for shape in self.shapes_to_draw:
            if shape["type"] == "rectangle":
                rect = shape["rect"]
                color = shape["color"]
                painter.setPen(QPen(QColor(*color), 3)) # 太さ3のペン
                painter.drawRect(rect[0], rect[1], rect[2], rect[3])

    def update_drawing(self, shapes):
        """ ワーカースレッドから描画情報を受け取るスロット """
        self.shapes_to_draw = shapes
        self.update_geometry() # ウィンドウ位置を再調整
        if not self.isVisible(): self.show() # 非表示なら表示する
        self.update() # 再描画をトリガー（paintEventが呼ばれる）


# --- メイン処理 ---
if __name__ == "__main__":
    # テンプレートフォルダの存在チェック
    if not os.path.exists(TEMPLATE_ROOT_DIR):
        print(f"エラー: '{TEMPLATE_ROOT_DIR}' フォルダが見つかりません。")
        exit()

    app = QApplication(sys.argv)
    window = OverlayWindow(interval=WORKER_INTERVAL)
    window.show()
    sys.exit(app.exec())