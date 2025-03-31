from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Step 1: Custom Dataset
class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        code = self.codes[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Step 2: Load and Preprocess Data
def load_data(db_name="code_data.db"):
    import sqlite3
    conn = sqlite3.connect(db_name)
    query = "SELECT * FROM CodeSnippets"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Step 3: Train and Test Split
def preprocess_data(df, tokenizer, max_length, batch_size):
    codes = df["code"].tolist()
    labels = [1 if label == "buggy" else 0 for label in df["label"]]
    train_codes, test_codes, train_labels, test_labels = train_test_split(
        codes, labels, test_size=0.2, random_state=42
    )
    train_dataset = CodeDataset(train_codes, train_labels, tokenizer, max_length)
    test_dataset = CodeDataset(test_codes, test_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

# Step 4: Training the Model
def train_model(model, train_loader, optimizer, criterion, device):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model after training is complete
    torch.save(model.state_dict(), "bug_detection_model.pt")
    print("Model has been successfully saved as bug_detection_model.pt")

# Step 5: Evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Main Function
if __name__ == "__main__":
    # Configuration
    max_length = 128
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print("Loading data...")
    df = load_data()

    # Tokenizer and Model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2
    )
    model.to(device)

    # Preprocess Data
    print("Preprocessing data...")
    train_loader, test_loader = preprocess_data(df, tokenizer, max_length, batch_size)

    # Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train Model
    print("Training model...")
    train_model(model, train_loader, optimizer, criterion, device)

    # Evaluate Model
    print("Evaluating model...")
    evaluate_model(model, test_loader, device)