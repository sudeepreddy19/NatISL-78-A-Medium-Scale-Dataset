import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)

# =========================================================
# CONFIG
# =========================================================

FEATURES_DIR = r"C:/Users/sudee/Desktop/BILSTM/features_by_word"

VISUAL_EMBEDDINGS_DIR = r"C:/Users/sudee/Desktop/bilstm/visual_embeddings"

TRAIN_PERSONS = [
    "person1",
    "person2",
    "person3",
    "person4",
    "person5"
]

TEST_PERSONS = [
    "person6"
]

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

VISUAL_DIM = 512
VISUAL_FEATURE_DIM = 128

TRAIN_VISUAL_SAMPLE_RATIO = 0.2
TEST_VISUAL_WEIGHT = 0.5

PATIENCE = 10
SEED = 42

MODEL_SAVE_PATH = "bilstm_attention_visual_best.pth"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using Device: {DEVICE}")

# =========================================================
# LOAD DATASET
# =========================================================

train_samples = []
test_samples = []

class_names = []
label_map = {}

print("\nLoading Dataset...\n")

valid_words = []

for word in sorted(os.listdir(FEATURES_DIR)):

    word_path = os.path.join(
        FEATURES_DIR,
        word
    )

    if not os.path.isdir(word_path):
        continue

    visual_embedding_path = os.path.join(
        VISUAL_EMBEDDINGS_DIR,
        f"{word}.npy"
    )

    if not os.path.exists(visual_embedding_path):
        print(f"Missing visual embedding: {word}")
        continue

    valid_words.append(word)

for label_idx, word in enumerate(valid_words):

    word_path = os.path.join(
        FEATURES_DIR,
        word
    )

    visual_embedding_path = os.path.join(
        VISUAL_EMBEDDINGS_DIR,
        f"{word}.npy"
    )

    class_names.append(word)
    label_map[word] = label_idx

    for person in sorted(os.listdir(word_path)):

        person_path = os.path.join(
            word_path,
            person
        )

        if not os.path.isdir(person_path):
            continue

        for file in sorted(os.listdir(person_path)):

            if not file.endswith(".npy"):
                continue

            feature_path = os.path.join(
                person_path,
                file
            )

            sample = {
                "feature_path": feature_path,
                "visual_path": visual_embedding_path,
                "label": label_idx,
                "word": word,
                "person": person,
                "visual_weight": 0.0
            }

            if person in TRAIN_PERSONS:
                train_samples.append(sample)

            elif person in TEST_PERSONS:
                sample["visual_weight"] = TEST_VISUAL_WEIGHT
                test_samples.append(sample)

# =========================================================
# ASSIGN 50% VISUAL CUES PER TRAINING PERSON
# =========================================================

for person in TRAIN_PERSONS:

    person_indices = [
        i for i, sample in enumerate(train_samples)
        if sample["person"] == person
    ]

    random.shuffle(person_indices)

    visual_count = int(
        len(person_indices) * TRAIN_VISUAL_SAMPLE_RATIO
    )

    visual_indices = set(
        person_indices[:visual_count]
    )

    for i in person_indices:

        if i in visual_indices:
            train_samples[i]["visual_weight"] = 1.0
        else:
            train_samples[i]["visual_weight"] = 0.0

print(f"Train Samples: {len(train_samples)}")
print(f"Test Samples: {len(test_samples)}")
print(f"Total Classes: {len(class_names)}")

print("\nTraining visual cue usage:")

for person in TRAIN_PERSONS:

    person_samples = [
        sample for sample in train_samples
        if sample["person"] == person
    ]

    visual_on = sum(
        1 for sample in person_samples
        if sample["visual_weight"] > 0
    )

    print(
        f"{person}: "
        f"{visual_on}/{len(person_samples)} samples with visual cue"
    )

print(
    f"\nTest visual cue weight: "
    f"{TEST_VISUAL_WEIGHT}"
)

if len(train_samples) == 0:
    raise ValueError("No training samples found.")

if len(test_samples) == 0:
    raise ValueError("No test samples found.")

# =========================================================
# DATASET
# =========================================================

class MultimodalDataset(Dataset):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        sign_features = np.load(
            sample["feature_path"]
        )

        visual_embedding = np.load(
            sample["visual_path"]
        )

        sign_features = torch.tensor(
            sign_features,
            dtype=torch.float32
        )

        visual_embedding = torch.tensor(
            visual_embedding,
            dtype=torch.float32
        )

        visual_weight = torch.tensor(
            sample["visual_weight"],
            dtype=torch.float32
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return (
            sign_features,
            visual_embedding,
            visual_weight,
            label
        )

# =========================================================
# COLLATE FUNCTION
# =========================================================

def collate_fn(batch):

    sequences = [
        item[0]
        for item in batch
    ]

    visual_embeddings = torch.stack([
        item[1]
        for item in batch
    ])

    visual_weights = torch.stack([
        item[2]
        for item in batch
    ])

    labels = torch.stack([
        item[3]
        for item in batch
    ])

    lengths = torch.tensor(
        [
            len(seq)
            for seq in sequences
        ],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True
    )

    return (
        padded_sequences,
        lengths,
        visual_embeddings,
        visual_weights,
        labels
    )

# =========================================================
# DATALOADERS
# =========================================================

train_dataset = MultimodalDataset(
    train_samples
)

test_dataset = MultimodalDataset(
    test_samples
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================================================
# ATTENTION
# =========================================================

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(
        self,
        lstm_output,
        mask
    ):

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
            lstm_output * weights.unsqueeze(-1),
            dim=1
        )

        return context

# =========================================================
# MULTIMODAL MODEL
# =========================================================

class MultimodalBiLSTM(nn.Module):

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
            dropout=DROPOUT if num_layers > 1 else 0
        )

        self.attention = Attention(
            hidden_size
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(
                VISUAL_DIM,
                VISUAL_FEATURE_DIM
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        fusion_dim = (
            hidden_size * 2
            + VISUAL_FEATURE_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                fusion_dim,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        x,
        lengths,
        visual_embedding,
        visual_weight
    ):

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
                lstm_out.size(1),
                device=lengths.device
            )[None, :]
            < lengths[:, None]
        )

        temporal_feature = self.attention(
            lstm_out,
            mask
        )

        visual_feature = self.visual_fc(
            visual_embedding
        )

        visual_feature = (
            visual_feature *
            visual_weight.unsqueeze(1)
        )

        fused = torch.cat(
            [
                temporal_feature,
                visual_feature
            ],
            dim=1
        )

        out = self.classifier(
            fused
        )

        return out

# =========================================================
# MODEL
# =========================================================

sample_feature = np.load(
    train_samples[0]["feature_path"]
)

INPUT_SIZE = sample_feature.shape[1]

NUM_CLASSES = len(class_names)

model = MultimodalBiLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )
)

# =========================================================
# TRAINING
# =========================================================

train_losses = []
test_losses = []

train_accs = []
test_accs = []

best_acc = 0.0
early_stop_counter = 0

best_y_true = []
best_y_pred = []

print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for (
        x,
        lengths,
        visual_embeddings,
        visual_weights,
        labels
    ) in train_loader:

        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)

        visual_embeddings = visual_embeddings.to(DEVICE)
        visual_weights = visual_weights.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            x,
            lengths,
            visual_embeddings,
            visual_weights
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

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

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # =====================================================
    # TEST / VALIDATION
    # =====================================================

    model.eval()

    total_test_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for (
            x,
            lengths,
            visual_embeddings,
            visual_weights,
            labels
        ) in test_loader:

            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)

            visual_embeddings = visual_embeddings.to(DEVICE)
            visual_weights = visual_weights.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(
                x,
                lengths,
                visual_embeddings,
                visual_weights
            )

            loss = criterion(
                outputs,
                labels
            )

            total_test_loss += loss.item()

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

    test_loss = total_test_loss / len(test_loader)
    test_acc = correct / total

    scheduler.step(test_acc)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if test_acc > best_acc:

        best_acc = test_acc

        best_y_true = y_true.copy()
        best_y_pred = y_pred.copy()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "label_map": label_map,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "visual_dim": VISUAL_DIM,
                "visual_feature_dim": VISUAL_FEATURE_DIM,
                "train_persons": TRAIN_PERSONS,
                "test_persons": TEST_PERSONS,
                "train_visual_sample_ratio": TRAIN_VISUAL_SAMPLE_RATIO,
                "test_visual_weight": TEST_VISUAL_WEIGHT,
                "best_acc": best_acc
            },
            MODEL_SAVE_PATH
        )

        print("Best model saved")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping")
        break

# =========================================================
# FINAL RESULTS
# =========================================================

if len(best_y_true) > 0:
    y_true = best_y_true
    y_pred = best_y_pred

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, axs = plt.subplots(
    1,
    3,
    figsize=(24, 7)
)

# =========================================================
# LOSS CURVE
# =========================================================

axs[0].plot(
    train_losses,
    label="Train Loss"
)

axs[0].plot(
    test_losses,
    label="Test Loss"
)

axs[0].set_title(
    "Loss Curve"
)

axs[0].set_xlabel(
    "Epoch"
)

axs[0].set_ylabel(
    "Loss"
)

axs[0].legend()

# =========================================================
# ACCURACY CURVE
# =========================================================

axs[1].plot(
    train_accs,
    label="Train Accuracy"
)

axs[1].plot(
    test_accs,
    label="Test Accuracy"
)

axs[1].set_title(
    "Accuracy Curve"
)

axs[1].set_xlabel(
    "Epoch"
)

axs[1].set_ylabel(
    "Accuracy"
)

axs[1].legend()

# =========================================================
# CONFUSION MATRIX
# =========================================================

sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    xticklabels=class_names,
    yticklabels=class_names,
    ax=axs[2]
)

axs[2].set_title(
    "Confusion Matrix"
)

axs[2].set_xlabel(
    "Predicted"
)

axs[2].set_ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "training_results.png",
    dpi=300
)

plt.show()

# =========================================================
# REPORT
# =========================================================

print("\n=================================================")
print("BEST TEST ACCURACY")
print("=================================================")

print(best_acc)

print("\n=================================================")
print("CLASSIFICATION REPORT")
print("=================================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

print(
    f"\nBest model saved as: "
    f"{MODEL_SAVE_PATH}"
)

print(
    "\nTraining graph saved as: "
    "training_results.png"
)