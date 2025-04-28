import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import os
import joblib  # For saving the LabelEncoder
import matplotlib.pyplot as plt  # <-- For plotting loss graph

# Load your CSV dataset
csv_path = r"D:\projectfilehandling\venv\finalyearproject\test_case_main_final.csv"
df = pd.read_csv(csv_path, encoding='cp1252')

# Assuming your CSV has columns "Text" and "Category"
texts = df["Text"].tolist()
labels_str = df["Category"].tolist()

# Use LabelEncoder to convert string labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels_str)

# Custom dataset class for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Training function with loss tracking and plotting
def train_and_save_model(train_dataset, model, tokenizer, optimizer, save_directory, num_epochs=36, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []  # To store average loss per epoch

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)  # Save average loss for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    # Save the entire model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    label_encoder_file = os.path.join(save_directory, 'label_encoder.joblib')
    joblib.dump(label_encoder, label_encoder_file)
    print(f"Model, tokenizer, and label encoder saved to {save_directory}")

    # Plotting Epochs vs Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.grid(True)
    plt.show()

    # Optional: Save the plot image
    plot_path = os.path.join(save_directory, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

# Example usage
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = len(df["Category"].unique())
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

train_dataset = TextClassificationDataset(texts, labels, tokenizer)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Specify your desired save directory
save_directory = r"D:\projectfilehandling\venv\saved_model"
train_and_save_model(train_dataset, model, tokenizer, optimizer, save_directory, num_epochs=36, batch_size=32)
