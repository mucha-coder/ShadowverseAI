import sys
import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
import pygetwindow as gw
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

# --- グローバル設定 ---
TEMPLATE_ROOT_DIR = "templates"
GAME_WINDOW_TITLE = "ShadowverseWB"
# ワーカースレッドの処理間隔（秒）
WORKER_INTERVAL = 2.0

# --- ワーカースレッド：重い処理を担当 ---
class Worker(QThread):
    # カスタムシグナル：描画すべき図形のリストを渡す
    calculation_finished = pyqtSignal(list)

    def ocr_read_current_pp(self, image: np.ndarray) -> int:
        """ PP画像から現在のPPの値のみを読み取るヘルパー関数 """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            if not text:
                return -1

            # PPが10の場合（"1010"など）とそれ以外で処理を分ける
            if len(text) == 4 and text.startswith("10"):
                current_pp_str = text[:2] # 最初の2文字 "10"
            else:
                # 文字列のちょうど半分までを取得
                half_len = len(text) // 2
                current_pp_str = text[:half_len]

            return int(current_pp_str)

        except (ValueError, TypeError):
            return -1

    def ocr_read_number(self, image: np.ndarray) -> int:
        """ 画像から数字を読み取るヘルパー関数 """
        try:
            # 画像の前処理（OCR精度向上のため）
            # グレースケール化 -> 二値化
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Tesseractの設定：対象を0-9の数字に限定
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"

            text = pytesseract.image_to_string(thresh, config=config)
            return int(text.strip())
        except (ValueError, TypeError):
            # 読み取れなかった、または数字に変換できなかった場合
            return -1 # エラーを示す値

    def find_elements(self, main_image, template_image, threshold=0.8):
        """ テンプレートマッチングを行い、見つかった要素の矩形リストを返す """
        result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        h, w = template_image.shape[:2]
        rects = []
        for pt in zip(*locations[::-1]):
            rects.append((pt[0], pt[1], w, h))
        return rects

    def run(self):
        """ スレッドのメイン処理 """
        while True:
            print("ワーカースレッドのループ実行中...") # ループが動いているか確認用
            # --- 1. 画像認識 ---
            try:
                game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
                x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
                screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
                main_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"ゲームウィンドウのキャプチャに失敗: {e}")
                time.sleep(self.parent().interval)
                continue

            # 全カテゴリのテンプレートマッチングを実行
            found_elements = {}
            for category in os.listdir(TEMPLATE_ROOT_DIR):
                category_path = os.path.join(TEMPLATE_ROOT_DIR, category)
                if not os.path.isdir(category_path): continue

                found_elements[category] = []
                for template_filename in os.listdir(category_path):
                    if not template_filename.endswith(".png"): continue
                    template_path = os.path.join(category_path, template_filename)
                    img_array = np.fromfile(template_path, np.uint8)
                    template_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if template_image is None: continue
                    
                    # find_elementsメソッドを使用して矩形リストを取得
                    rects = self.find_elements(main_image, template_image, threshold=0.8)
                    if rects:
                        found_elements[category].extend(rects)

            # --- 2. カード位置の推定 ---
            # costテンプレートのマッチング結果から手札のおおよその位置を計算する
            estimated_hand_cards = []
            if 'cost' in found_elements and found_elements['cost']:
                sorted_costs = sorted(found_elements['cost'], key=lambda item: item[0])
                
                # オフセット値とカードサイズ（ここはご自身の環境に合わせて調整が必要）
                OFFSET_X = 15
                OFFSET_Y = 15
                CARD_WIDTH = 100
                CARD_HEIGHT = 100

                for (x, y, w, h) in sorted_costs:
                    card_x = x - OFFSET_X
                    card_y = y - OFFSET_Y
                    estimated_hand_cards.append((card_x, card_y, CARD_WIDTH, CARD_HEIGHT))

            # --- 3. AI思考（高度化版） ---
            action_plan = {"highlight_rects": []}

            print("AI思考開始")

            # まず、現在のPPを読み取る
            current_pp = -1
            if 'my_pp' in found_elements and found_elements['my_pp']:
                x, y, w, h = found_elements['my_pp'][0]
                pp_image = main_image[y:y+h, x:x+w]
                current_pp = self.ocr_read_current_pp(pp_image)
                print(f"認識したPP: {current_pp}")

            # PPが正常に読み取れ、かつ手札カードが推定できている場合のみ思考する
            if current_pp != -1 and estimated_hand_cards:
                playable_cards = []
                
                # コスト部分を正確に切り出すためのオフセット（カード左上からの相対位置）
                COST_OFFSET_X = 5   # 要調整
                COST_OFFSET_Y = 5   # 要調整
                COST_AREA_W = 30    # コスト部分の幅（要調整）
                COST_AREA_H = 30    # コスト部分の高さ（要調整）

                for (card_x, card_y, card_w, card_h) in estimated_hand_cards:
                    cost_area_x_abs = card_x + COST_OFFSET_X
                    cost_area_y_abs = card_y + COST_OFFSET_Y
                    cost_image = main_image[cost_area_y_abs : cost_area_y_abs + COST_AREA_H, 
                                            cost_area_x_abs : cost_area_x_abs + COST_AREA_W]
                    
                    card_cost = self.ocr_read_number(cost_image)
                    print(f"カード位置({card_x}, {card_y}) -> 認識したコスト: {card_cost}")

                    if 0 < card_cost <= current_pp:
                        playable_cards.append({
                            "rect": (card_x, card_y, card_w, card_h),
                            "cost": card_cost
                        })

                # プレイ可能なカードがあれば、その中で最もコストが高いものを選択
                if playable_cards:
                    best_card = max(playable_cards, key=lambda c: c["cost"])
                    action_plan["highlight_rects"].append(best_card["rect"])
                    print(f"プレイ候補: コスト {best_card['cost']} のカード")
            
            # --- 4. 描画情報の作成 ---
            shapes_to_draw = []
            for (x, y, w, h) in action_plan["highlight_rects"]:
                shapes_to_draw.append({
                    "type": "rectangle",
                    "rect": (x, y, w, h),
                    "color": (255, 0, 0, 200) # RGBA
                })

            # --- 5. UIスレッドへ結果を通知 ---
            self.calculation_finished.emit(shapes_to_draw)

            # 指定した間隔で待機
            time.sleep(self.parent().interval)


# --- UIスレッド：オーバーレイウィンドウの描画を担当 ---
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