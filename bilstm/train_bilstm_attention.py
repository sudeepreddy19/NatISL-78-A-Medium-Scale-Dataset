import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)

# ==================================================
# CONFIG
# ==================================================

DATASET_DIR = r"C:/Users/sudee/Desktop/BILSTM/features_by_word1"

BATCH_SIZE = 16
EPOCHS = 80
LR = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "bilstm_attention_best.pth"

# ==================================================
# LOAD DATA
# ==================================================

X = []
y = []
class_names = []
label_map = {}

print("Loading dataset...")

for label_idx, word in enumerate(sorted(os.listdir(DATASET_DIR))):

    word_path = os.path.join(DATASET_DIR, word)

    if not os.path.isdir(word_path):
        continue

    label_map[word] = label_idx
    class_names.append(word)

    for person in os.listdir(word_path):

        person_path = os.path.join(word_path, person)

        if not os.path.isdir(person_path):
            continue

        for file in os.listdir(person_path):

            if file.endswith(".npy"):

                file_path = os.path.join(person_path, file)

                features = np.load(file_path)

                if len(features) > 0:
                    X.append(features)
                    y.append(label_idx)

print(f"Total Samples: {len(X)}")

# ==================================================
# TRAIN / VAL SPLIT
# ==================================================

indices = np.arange(len(X))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train = [X[i] for i in train_idx]
X_val = [X[i] for i in val_idx]

y_train = [y[i] for i in train_idx]
y_val = [y[i] for i in val_idx]

# ==================================================
# DATASET
# ==================================================

class SignDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )

        return x, y

# ==================================================
# COLLATE FUNCTION
# ==================================================

def collate_fn(batch):

    sequences = [item[0] for item in batch]
    labels = torch.tensor(
        [item[1] for item in batch]
    )

    lengths = torch.tensor(
        [len(seq) for seq in sequences]
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True
    )

    return padded_sequences, lengths, labels

train_dataset = SignDataset(
    X_train,
    y_train
)

val_dataset = SignDataset(
    X_val,
    y_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ==================================================
# ATTENTION
# ==================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, lstm_output, mask):

        scores = self.attention(
            lstm_output
        ).squeeze(-1)

        scores = scores.masked_fill(
            mask == 0,
            -1e9
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            lstm_output *
            weights.unsqueeze(-1),
            dim=1
        )

        return context

# ==================================================
# BiLSTM + ATTENTION
# ==================================================

class BiLSTMAttention(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )

        self.attention = Attention(
            hidden_size
        )

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x, lengths):

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(
            packed
        )

        lstm_out, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        mask = (
            torch.arange(
                lstm_out.size(1)
            )[None, :]
            .to(lengths.device)
            < lengths[:, None]
        )

        context = self.attention(
            lstm_out,
            mask
        )

        out = self.fc(context)

        return out

# ==================================================
# MODEL
# ==================================================

input_size = X[0].shape[1]
num_classes = len(class_names)

model = BiLSTMAttention(
    input_size,
    HIDDEN_SIZE,
    NUM_LAYERS,
    num_classes
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ==================================================
# TRAINING
# ==================================================

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_acc = 0

for epoch in range(EPOCHS):

    # ================= TRAIN =================
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, lengths, labels in train_loader:

        x = x.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            x,
            lengths
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        total_loss /
        len(train_loader)
    )

    train_acc = (
        correct / total
    )

    # ================= VALIDATION =================
    model.eval()

    val_loss_total = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, lengths, labels in val_loader:

            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                x,
                lengths
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    val_loss = (
        val_loss_total /
        len(val_loader)
    )

    val_acc = (
        correct / total
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            "✅ Best model saved"
        )

# ==================================================
# CURVES
# ==================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses,
         label='Train Loss')
plt.plot(val_losses,
         label='Val Loss')
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs,
         label='Train Accuracy')
plt.plot(val_accs,
         label='Val Accuracy')
plt.title("Accuracy Curve")
plt.legend()

plt.show()

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(12,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)