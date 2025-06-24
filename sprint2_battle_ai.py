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
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF
import pytesseract

# --- Tesseract OCRのパス設定 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- グローバル設定 & 調整が必要な定数 ---
# -------------------------------------------------------------
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 以下のオフセット値やサイズは、あなたのゲーム解像度に合わせて調整が必要です
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
CARD_ATTACK_OFFSET = {"x": 20, "y": 145, "w": 45, "h": 45}  # カード左上からの攻撃力数字の相対位置
CARD_HEALTH_OFFSET = {"x": 95, "y": 145, "w": 45, "h": 45}  # カード左上からの体力数字の相対位置
# -------------------------------------------------------------
GAME_WINDOW_TITLE = "Shadowverse"
TEMPLATE_ROOT_DIR = "templates"
WORKER_INTERVAL = 2.0

class Worker(QThread):
    calculation_finished = pyqtSignal(list)

    # ... ocr_read_number, find_elements 関数は前回から変更なし ...
    def ocr_read_number(self, image: np.ndarray) -> int:
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
        # TODO: 重複領域を削除するロジック(NMS)が必要
        for pt in zip(*locations[::-1]):
            found_points.append((pt[0], pt[1], w, h))
        return found_points

    def recognize_followers(self, main_image, category):
        """ [新規] 場のフォロワーを認識する関数 """
        followers = []
        category_path = os.path.join(TEMPLATE_ROOT_DIR, category)
        if not os.path.isdir(category_path): return followers

        for template_filename in os.listdir(category_path):
            template_path = os.path.join(category_path, template_filename)
            template_image = cv2.imread(template_path)
            if template_image is None: continue

            follower_locations = self.find_elements(main_image, template_image, threshold=0.85)

            for (x, y, w, h) in follower_locations:
                # 攻撃力を読み取る
                atk_off = CARD_ATTACK_OFFSET
                atk_img = main_image[y+atk_off['y']:y+atk_off['y']+atk_off['h'], x+atk_off['x']:x+atk_off['x']+atk_off['w']]
                attack = self.ocr_read_number(atk_img)
                
                # 体力を読み取る
                hel_off = CARD_HEALTH_OFFSET
                hel_img = main_image[y+hel_off['y']:y+hel_off['y']+hel_off['h'], x+hel_off['x']:x+hel_off['x']+hel_off['w']]
                health = self.ocr_read_number(hel_img)

                if attack != -1 and health != -1:
                    followers.append({"rect": (x, y, w, h), "attack": attack, "health": health})
        return followers
    
    def decide_attack_plan(self, my_followers, enemy_followers):
        """ [新規] 有利トレードを判断するAIロジック """
        attack_plan = []
        
        # 攻撃済みの相手や行動済みの自分を管理するリスト
        attacked_enemies = []
        acted_allies = []

        # 優先度1: 有利トレードを探す
        for i, my_f in enumerate(my_followers):
            for j, enemy_f in enumerate(enemy_followers):
                # 既に行動/攻撃済みの場合はスキップ
                if i in acted_allies or j in attacked_enemies:
                    continue
                
                # 有利トレードの条件
                is_favorable_trade = (my_f['attack'] >= enemy_f['health'] and my_f['health'] > enemy_f['attack'])
                if is_favorable_trade:
                    attack_plan.append({"from": my_f, "to": enemy_f})
                    acted_allies.append(i)
                    attacked_enemies.append(j)
                    break # この自分のフォロワーは行動済み

        # 優先度2: 残ったフォロワーで相手リーダーを攻撃
        for i, my_f in enumerate(my_followers):
            if i not in acted_allies:
                attack_plan.append({"from": my_f, "to": "enemy_leader"})
                
        return attack_plan


    def run(self):
        while True:
            time.sleep(self.parent().interval)
            try:
                game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
                x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
                main_image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
            
            # --- 1. 盤面認識 ---
            my_followers = self.recognize_followers(main_image, 'my_board_card')
            enemy_followers = self.recognize_followers(main_image, 'enemy_board_card')
            
            # --- 2. 攻撃判断 ---
            attack_plan = self.decide_attack_plan(my_followers, enemy_followers)
            
            # --- 3. 手札プレイ判断 (既存のロジック) ---
            # ... (PPと手札コストを読み取り、プレイすべきカードを決めるロジック) ...
            best_play_card = None # 仮

            # --- 4. 描画情報を作成 ---
            shapes_to_draw = []
            # 攻撃プランを矢印として追加
            for attack in attack_plan:
                my_f_rect = attack['from']['rect']
                start_point = (my_f_rect[0] + my_f_rect[2] // 2, my_f_rect[1] + my_f_rect[3] // 2)
                
                if attack['to'] == 'enemy_leader':
                    # TODO: 相手リーダーの座標を特定する
                    end_point = (main_image.shape[1] // 2, 100)
                else:
                    enemy_f_rect = attack['to']['rect']
                    end_point = (enemy_f_rect[0] + enemy_f_rect[2] // 2, enemy_f_rect[1] + enemy_f_rect[3] // 2)
                
                shapes_to_draw.append({"type": "arrow", "start": start_point, "end": end_point, "color": (255, 165, 0, 255)})
            
            # 手札プレイを四角として追加
            if best_play_card:
                shapes_to_draw.append({"type": "rectangle", "rect": best_play_card['rect'], "color": (0, 255, 0, 220)})

            self.calculation_finished.emit(shapes_to_draw)

class OverlayWindow(QMainWindow):
    # ... (前回から変更なし、ただしpaintEventを更新) ...
    def __init__(self, interval=2.0):
        super().__init__()
        self.interval = interval
        self.shapes_to_draw = []
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.update_geometry()
        self.worker = Worker()
        self.worker.setParent(self)
        self.worker.calculation_finished.connect(self.update_drawing)
        self.worker.start()

    def update_geometry(self):
        try:
            game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
            self.setGeometry(game_window.left, game_window.top, game_window.width, game_window.height)
        except Exception:
            self.hide()

    def paintEvent(self, event):
        """ [更新] 矢印描画機能を追加 """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing) # 線を滑らかにする

        for shape in self.shapes_to_draw:
            color = QColor(*shape.get("color", (255, 0, 0, 255)))
            pen = QPen(color, 4, Qt.PenStyle.SolidLine) # 太さ4のペン
            painter.setPen(pen)
            
            if shape["type"] == "rectangle":
                rect = shape["rect"]
                painter.drawRect(rect[0], rect[1], rect[2], rect[3])
            
            elif shape["type"] == "arrow":
                start_p = QPointF(*shape["start"])
                end_p = QPointF(*shape["end"])
                painter.drawLine(start_p, end_p)
                
                # 矢じりを描画
                angle = math.atan2(start_p.y() - end_p.y(), start_p.x() - end_p.x())
                arrow_size = 15.0
                arrow_p1 = end_p + QPointF(math.sin(angle + math.pi / 3) * arrow_size, math.cos(angle + math.pi / 3) * arrow_size)
                arrow_p2 = end_p + QPointF(math.sin(angle + math.pi - math.pi / 3) * arrow_size, math.cos(angle + math.pi - math.pi / 3) * arrow_size)
                arrow_head = QPolygonF([end_p, arrow_p1, arrow_p2])
                painter.setBrush(color)
                painter.drawPolygon(arrow_head)

    def update_drawing(self, shapes):
        self.shapes_to_draw = shapes
        self.update_geometry()
        if not self.isVisible(): self.show()
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayWindow(interval=WORKER_INTERVAL)
    window.show()
    sys.exit(app.exec())