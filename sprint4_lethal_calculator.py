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

# --- Tesseract OCRのパス設定 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- グローバル設定 & 調整が必要な定数 ---
CARD_ATTACK_OFFSET = {"x": 20, "y": 145, "w": 45, "h": 45}; CARD_HEALTH_OFFSET = {"x": 95, "y": 145, "w": 45, "h": 45}
HAND_CARD_COST_OFFSET = {"x": 15, "y": 15, "w": 40, "h": 40}; HAND_CARD_SIZE = {"w": 160, "h": 210}
GAME_WINDOW_TITLE = "Shadowverse"; TEMPLATE_ROOT_DIR = "templates"; WORKER_INTERVAL = 2.0

class Worker(QThread):
    calculation_finished = pyqtSignal(list)

    # --- 認識ヘルパー関数群 (ocr_read_number, find_elementsは変更なし) ---
    def ocr_read_number(self, image: np.ndarray) -> int:
        # ... (変更なし)
        try:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);sf=3;rs=cv2.resize(gray,(gray.shape[1]*sf,gray.shape[0]*sf),interpolation=cv2.INTER_CUBIC);_,th=cv2.threshold(rs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU);conf="--psm 7 -c tessedit_char_whitelist=0123456789";txt=pytesseract.image_to_string(th,config=conf,lang='eng');return int(txt.strip())
        except: return -1

    def find_elements(self, main_image, template_image, threshold=0.8):
        # ... (変更なし)
        h,w=template_image.shape[:2];res=cv2.matchTemplate(main_image,template_image,cv2.TM_CCOEFF_NORMED);loc=np.where(res>=threshold);pts=[];[pts.append((pt[0],pt[1],w,h)) for pt in zip(*loc[::-1])];return pts

    def recognize_followers(self, main_image, category, shugo_template):
        # ... (前回から変更なし)
        followers=[];category_path=os.path.join(TEMPLATE_ROOT_DIR,category);
        if not os.path.isdir(category_path) or shugo_template is None:return followers
        for fname in os.listdir(category_path):
            t_img=cv2.imread(os.path.join(category_path,fname));
            if t_img is None:continue
            for x,y,w,h in self.find_elements(main_image,t_img,threshold=0.85):
                atk_img=main_image[y+CARD_ATTACK_OFFSET['y']:y+CARD_ATTACK_OFFSET['y']+CARD_ATTACK_OFFSET['h'],x+CARD_ATTACK_OFFSET['x']:x+CARD_ATTACK_OFFSET['x']+CARD_ATTACK_OFFSET['w']];atk=self.ocr_read_number(atk_img)
                hel_img=main_image[y+CARD_HEALTH_OFFSET['y']:y+CARD_HEALTH_OFFSET['y']+CARD_HEALTH_OFFSET['h'],x+CARD_HEALTH_OFFSET['x']:x+CARD_HEALTH_OFFSET['x']+CARD_HEALTH_OFFSET['w']];hel=self.ocr_read_number(hel_img)
                abil=[];f_area=main_image[y:y+h,x:x+w]
                if self.find_elements(f_area,shugo_template,threshold=0.8):abil.append("shugo")
                if atk!=-1 and hel!=-1:followers.append({"rect":(x,y,w,h),"attack":atk,"health":hel,"abilities":abil})
        return followers
    
    def recognize_hand_cards(self, main_image, cost_templates, shissou_template):
        """ [新規] 手札のカード情報（コスト、疾走有無）を認識する """
        hand_cards = []
        if shissou_template is None: return hand_cards
        for cost_str, cost_template in cost_templates.items():
            card_cost = int(cost_str)
            for x,y,w,h in self.find_elements(main_image, cost_template, threshold=0.9):
                card_rect = (x - HAND_CARD_COST_OFFSET['x'], y - HAND_CARD_COST_OFFSET['y'], HAND_CARD_SIZE['w'], HAND_CARD_SIZE['h'])
                card_area_image = main_image[card_rect[1]:card_rect[1]+card_rect[3], card_rect[0]:card_rect[0]+card_rect[2]]
                
                abilities = []
                if self.find_elements(card_area_image, shissou_template, threshold=0.8):
                    abilities.append("shissou")
                
                hand_cards.append({"rect": card_rect, "cost": card_cost, "attack": -1, "abilities": abilities}) # attackは後で読み取る
        return hand_cards

    def run(self):
        # --- テンプレートの事前読み込み ---
        shugo_template = cv2.imread(os.path.join(TEMPLATE_ROOT_DIR, 'ability', 'shugo_icon.png'))
        shissou_template = cv2.imread(os.path.join(TEMPLATE_ROOT_DIR, 'ability', 'shissou_icon.png'))
        enemy_health_template = cv2.imread(os.path.join(TEMPLATE_ROOT_DIR, 'enemy_leader', 'health_area.png'))
        cost_templates = {fname.replace('.png','').split('_')[-1]: cv2.imread(os.path.join(TEMPLATE_ROOT_DIR, 'cost', fname)) for fname in os.listdir(os.path.join(TEMPLATE_ROOT_DIR,'cost'))}

        while True:
            time.sleep(self.parent().interval)
            try:
                # ... (画面キャプチャ) ...
                game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]; x1, y1, width, height = game_window.left, game_window.top, game_window.width, game_window.height
                main_image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))), cv2.COLOR_RGB_BGR)
            except Exception: continue

            # --- 1. 全情報の認識 ---
            my_followers = self.recognize_followers(main_image, 'my_board_card', shugo_template)
            enemy_followers = self.recognize_followers(main_image, 'enemy_board_card', shugo_template)
            hand_cards = self.recognize_hand_cards(main_image, cost_templates, shissou_template)
            
            current_pp = -1 # ... (PP認識ロジック) ...
            enemy_health = -1
            if enemy_health_template is not None:
                health_areas = self.find_elements(main_image, enemy_health_template, 0.85)
                if health_areas:
                    x,y,w,h = health_areas[0]; enemy_health = self.ocr_read_number(main_image[y:y+h, x:x+w])

            # --- 2. リーサル計算 ---
            total_damage = 0
            lethal_plan_shapes = []
            is_lethal = False
            has_guard = any("shugo" in f.get("abilities", []) for f in enemy_followers)

            if not has_guard and enemy_health != -1:
                # 場のフォロワーの攻撃力
                board_damage = sum(f['attack'] for f in my_followers)
                total_damage += board_damage
                for f in my_followers:
                    start_p = (f['rect'][0] + f['rect'][2]//2, f['rect'][1] + f['rect'][3]//2)
                    end_p = (main_image.shape[1] // 2, 100) # 相手リーダー位置
                    lethal_plan_shapes.append({"type":"arrow", "start":start_p, "end":end_p, "color":(255, 69, 0, 255)})
                
                # 手札の疾走フォロワーの攻撃力
                # TODO: プレイ順とPP消費を考慮した、より高度な計算が必要
                for card in hand_cards:
                    if "shissou" in card.get("abilities", []) and card['cost'] <= current_pp:
                        # 疾走フォロワーの攻撃力をOCRで読み取る
                        # ...
                        # total_damage += shissou_attack
                        pass
                
                if total_damage >= enemy_health:
                    is_lethal = True
            
            # --- 3. 描画情報を作成 ---
            shapes_to_draw = []
            if is_lethal:
                shapes_to_draw = lethal_plan_shapes
                # LETHALテキストを追加
                shapes_to_draw.append({"type":"text", "text":"LETHAL!!", 
                                       "pos":(main_image.shape[1]//2, main_image.shape[0]//2), 
                                       "font_size":100, "color":(255, 215, 0, 255)})
            else:
                # 通常の攻撃判断・手札プレイ判断
                # ... (sprint3_guard_logic.pyのロジックをここに) ...
                pass
            
            self.calculation_finished.emit(shapes_to_draw)

class OverlayWindow(QMainWindow):
    # ... (変更なし、ただしpaintEventを更新) ...
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for shape in self.shapes_to_draw:
            color = QColor(*shape.get("color", (255, 0, 0, 255))); pen = QPen(color, 4, Qt.PenStyle.SolidLine); painter.setPen(pen)
            shape_type = shape.get("type", "rectangle")
            if shape_type == "rectangle":
                rect = shape["rect"]; painter.drawRect(rect[0], rect[1], rect[2], rect[3])
            elif shape_type == "arrow":
                # ... (矢印描画ロジック) ...
            elif shape_type == "text":
                font = QFont("Arial", shape.get("font_size", 50), QFont.Weight.Bold)
                painter.setFont(font)
                pos = shape['pos']
                # テキストに影を付けて見やすくする
                shadow_pos = (pos[0]+3, pos[1]+3)
                painter.setPen(QColor(0,0,0,150)); painter.drawText(shadow_pos[0], shadow_pos[1], shape['text'])
                painter.setPen(color); painter.drawText(pos[0], pos[1], shape['text'])
    # ... (他のメソッドは変更なし) ...

# ... (if __name__ == "__main__": ブロックは変更なし) ...