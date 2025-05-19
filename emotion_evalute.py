import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re # 簡単なテキストクリーニング用

# サンプルデータ (テキストとラベル)
texts = [
    "this movie is great and amazing",
    "I really enjoyed this film",
    "what a terrible and awful movie",
    "this is a bad film, I hated it",
    "absolutely fantastic, loved every minute",
    "the worst movie I have ever seen"
]
labels = [1, 1, 0, 0, 1, 0] # 1: ポジティブ, 0: ネガティブ

# 簡単なテキストクリーニング関数
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # 句読点などを削除
    return text

cleaned_texts = [preprocess_text(text) for text in texts]

# トークン化 (スペースで区切るだけの簡単なもの)
tokenized_texts = [text.split() for text in cleaned_texts]
print("Tokenized Texts:", tokenized_texts)

# 語彙 (Vocabulary) の構築
word_counts = Counter()
for tokens in tokenized_texts:
    word_counts.update(tokens)

# 単語にIDを割り当てる (0はパディング用、1は未知語用に予約することが多い)
vocab = {"<PAD>": 0, "<UNK>": 1} # PAD: パディング, UNK: 未知語
for word, count in word_counts.items():
    if word not in vocab: # 頻度などでフィルタリングも可能
        vocab[word] = len(vocab)
print("Vocabulary:", vocab)
vocab_size = len(vocab)

# 数値化 (トークンをIDに変換)
numericalized_texts = []
for tokens in tokenized_texts:
    numericalized_texts.append([vocab.get(token, vocab["<UNK>"]) for token in tokens])
print("Numericalized Texts:", numericalized_texts)

# パディング (シーケンスの長さを揃える)
max_len = max(len(seq) for seq in numericalized_texts) # 最長のシーケンス長に合わせる

padded_texts = []
for seq in numericalized_texts:
    padding_needed = max_len - len(seq)
    padded_texts.append(seq + [vocab["<PAD>"]] * padding_needed)

print("Padded Texts:", padded_texts)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float) # BCEWithLogitsLoss用

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# データセットとデータローダーの作成
batch_size = 2 # 小さなデータセットなのでバッチサイズも小さく
train_dataset = TextDataset(padded_texts, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, dropout_p=0.2):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True, # 双方向LSTM
                            batch_first=True,   # 入力テンソルの最初の次元をバッチサイズにする
                            dropout=dropout_p if n_layers > 1 else 0) # n_layers > 1 の時のみ有効
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # 双方向なので hidden_dim * 2
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text_batch):
        # text_batch: (batch_size, seq_len)
        embedded = self.embedding(text_batch)
        # embedded: (batch_size, seq_len, embed_dim)

        embedded = self.dropout(embedded) # Embedding層の後にもDropoutを入れることがある

        # LSTMの出力は (output, (hidden_state, cell_state))
        # output: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden_state: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 最後の隠れ層の状態を使用 (双方向なので順方向と逆方向を連結)
        # hidden は (num_layers * 2, batch_size, hidden_dim) の形状
        # 最後の層の隠れ状態を取り出す
        # hidden[-2,:,:] は最後の層の順方向、hidden[-1,:,:] は最後の層の逆方向
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_concat: (batch_size, hidden_dim * 2)

        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        # output: (batch_size, output_dim)
        return output.squeeze(1) # output_dim=1の場合、(batch_size, 1) -> (batch_size)

# モデルのハイパーパラメータ
embed_dim = 100
hidden_dim = 64
output_dim = 1 # 二値分類 (ポジティブ/ネガティブの1つのスコア)
n_layers = 1   # LSTMの層数
dropout_p = 0.3

model = TextClassifier(vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout_p)
criterion = nn.BCEWithLogitsLoss() # 出力層にSigmoidがない場合に適している
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
epochs = 20 # 小さなデータセットなのでエポック数を多めに

for epoch in range(epochs):
    model.train() # 学習モード
    epoch_loss = 0
    epoch_acc = 0

    for texts_batch, labels_batch in train_loader:
        texts_batch = texts_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        predictions = model(texts_batch)

        loss = criterion(predictions, labels_batch)

        # 精度計算 (BCEWithLogitsLossなのでSigmoidを適用してから閾値で判定)
        predicted_classes = torch.round(torch.sigmoid(predictions))
        correct_predictions = (predicted_classes == labels_batch).float()
        acc = correct_predictions.sum() / len(correct_predictions)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_acc = epoch_acc / len(train_loader)
    print(f'Epoch {epoch+1:02}/{epochs} | Loss: {avg_epoch_loss:.4f} | Accuracy: {avg_epoch_acc*100:.2f}%')

# predict_sentiment 関数の定義を学習ループの外に移動し、インデントを修正
def predict_sentiment(text, model, vocab, max_len, device):
    model.eval() # 評価モード
    cleaned_text = preprocess_text(text)
    tokenized_text = cleaned_text.split()
    numericalized_text = [vocab.get(token, vocab["<UNK>"]) for token in tokenized_text]
    # パディング
    padding_needed = max_len - len(numericalized_text)
    if padding_needed < 0: # 長すぎる場合は切り詰める
        numericalized_text = numericalized_text[:max_len]
        padding_needed = 0
    padded_text = numericalized_text + [vocab["<PAD>"]] * padding_needed

    input_tensor = torch.tensor([padded_text], dtype=torch.long).to(device) # バッチサイズ1として入力

    with torch.no_grad(): # 勾配計算をオフに
        prediction = model(input_tensor)
        probability = torch.sigmoid(prediction).item()

    return "Positive" if probability > 0.5 else "Negative", probability

# テスト (ここもインデントなしで開始)
test_text1 = "This product is incredibly well-made and durable. Worth every penny."
sentiment1, prob1 = predict_sentiment(test_text1, model, vocab, max_len, device)
print(f"'{test_text1}' -> Sentiment: {sentiment1} (Prob: {prob1:.4f})")

test_text2 = "This useless product broke after just a few uses. Worst  quality I have eve seen.I hate this products."
sentiment2, prob2 = predict_sentiment(test_text2, model, vocab, max_len, device)
print(f"'{test_text2}' -> Sentiment: {sentiment2} (Prob: {prob2:.4f})")
