import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- グローバル設定 ---
LOG_FILENAME = "game_log.csv"
MODEL_SAVE_PATH = "imitation_model.pth"

# ベクトル次元数と行動の種類の数を定義（sprintX_data_logger.pyと合わせる）
VECTOR_SIZE = 81 # 仮。ご自身の設計に合わせてください
NUM_ACTIONS = 10 # 行動の種類（例：手札9枚プレイ + 何もしない）

# ハイパーパラメータ
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 1. データ準備 ---
def load_and_preprocess_data(filename):
    print("データの読み込みと前処理を開始...")
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"エラー: {filename} が見つかりません。まずデータ収集用のスクリプトを実行してください。")
        return None, None

    # 文字列形式のベクトルをNumpy配列に変換
    # ' [1., 2., ...]' -> [1., 2., ...]
    X_list = [np.fromstring(vec.replace('[', '').replace(']', ''), sep=',') for vec in df['board_vector']]
    X = np.array(X_list, dtype=np.float32)

    # 行動データをラベルに変換（今回は簡略化）
    # TODO: 本来はaction_dataのJSONを解析し、行動IDに変換する
    # 今回はダミーとして、0-9のランダムな行動IDを生成します
    y = np.random.randint(0, NUM_ACTIONS, size=len(df))
    y = np.array(y, dtype=np.int64)

    print("データ準備完了。")
    return X, y

# --- 2. モデルの定義 ---
class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
            # Softmaxは損失関数に含まれるため、ここでは不要
        )

    def forward(self, x):
        return self.network(x)

# --- 3. 訓練プロセス ---
def train_model(X, y):
    # データを訓練用と検証用に分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # PyTorchのデータセットとデータローダーを作成
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # モデル、損失関数、オプティマイザを定義
    model = ImitationModel(VECTOR_SIZE, NUM_ACTIONS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nモデルの訓練を開始します...")
    for epoch in range(EPOCHS):
        model.train() # 訓練モード
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # 順伝播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"エポック [{epoch+1}/{EPOCHS}], 損失: {avg_loss:.4f}")

    print("訓練が完了しました。")
    
    # --- 4. モデルの保存 ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")


if __name__ == "__main__":
    X_data, y_data = load_and_preprocess_data(LOG_FILENAME)
    if X_data is not None and y_data is not None:
        train_model(X_data, y_data)