import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

data_dir = "fake_training_data"
phrases = sorted(list(set([f.split("_")[1] for f in os.listdir(data_dir) if f.endswith(".wav")])))
label_map = {label: idx for idx, label in enumerate(phrases)}

# Initialize wav2vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

X, y = [], []

for f in os.listdir(data_dir):
    if not f.endswith(".wav"):
        continue
    file_path = os.path.join(data_dir, f)
    audio, sr = librosa.load(file_path, sr=16000)
    # Extract embedding
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        embedding = model(input_values).last_hidden_state.mean(dim=1).squeeze().numpy()
    X.append(embedding)
    
    # Extract label from filename (assumes format phrase_XX_YY.wav)
    label_idx = int(f.split("_")[1])
    y.append(label_idx)

X = np.stack(X)
y = np.array(y)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

import torch.nn as nn

class PhraseClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

import torch
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = PhraseClassifier(input_dim=X.shape[1], num_classes=25).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

epochs = 20

for epoch in range(epochs):
    classifier.train()
    running_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = classifier(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Validation
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = classifier(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    print(f"Validation Accuracy: {correct/total:.2%}")
