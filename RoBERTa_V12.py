import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
import random
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
seed = 35
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Load datasets
train_df = pd.read_csv('train_dataV3.csv')
validation_df = pd.read_csv('valid_LIAR2.csv')
test_df = pd.read_csv('test_LIAR2.csv')

# Load pre-trained RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = 6

# Tokenize and prepare dataset
def prepare_dataset(df, tokenizer):
    df = df.fillna('')
    df = df.astype(str)
    statement_encodings = tokenizer(df['statement'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    input_ids = statement_encodings['input_ids']
    attention_mask = statement_encodings['attention_mask']
    labels = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    labels = torch.tensor(labels.values)
    return TensorDataset(input_ids, attention_mask, labels)

# Prepare datasets
train_dataset = prepare_dataset(train_df, tokenizer)
val_dataset = prepare_dataset(validation_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)

# DataLoaders
batch_size = 108
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define model, optimizer, and scheduler
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.001, no_deprecation_warning=True)
num_epochs = 6
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return total_loss / len(data_loader), all_labels, all_preds

# Training loop with loss tracking and early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_losses = []

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
    val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the best model and training state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss
        }, 'best_model_checkpoint_V1.pth')
        print("Validation loss improved. Model and state saved.")
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
        
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered. Stopping training.")
        break

# Load the best model and state for final evaluation
checkpoint = torch.load('best_model_checkpoint_V1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
model.to(device)

# Plot training and validation loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_roberta_statement.png')
plt.show()

# Evaluate on training, validation, and test sets
def display_results(model, data_loader, dataset_name):
    loss, all_labels, all_preds = evaluate(model, data_loader, loss_fn, device)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean() * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Get classification report as a dictionary
    class_report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=[f"Class {i}" for i in range(num_labels)], 
        output_dict=True, 
        digits=2
    )
    
    # Get string version of classification report for display
    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=[f"Class {i}" for i in range(num_labels)], 
        digits=2
    )
    
    # Calculate per-label accuracy
    per_label_accuracy = {}
    per_label_f1 = {}
    for i, class_name in enumerate([f"Class {i}" for i in range(num_labels)]):
        true_positives = conf_matrix[i, i]
        total_actual = conf_matrix[i].sum()
        per_label_accuracy[class_name] = (true_positives / total_actual * 100) if total_actual > 0 else 0.0
        
        # Per-label F1 score
        precision = class_report_dict[class_name]["precision"]
        recall = class_report_dict[class_name]["recall"]
        per_label_f1[class_name] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Extract total F1 score (weighted average)
    total_f1_score = class_report_dict["weighted avg"]["f1-score"] * 100

    print(f"\n{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")
    print(f"{dataset_name} Multi-Class Confusion Matrix:")
    print(conf_matrix)
    print(f"{dataset_name} Classification Report:")
    print(class_report)

    # Print per-label accuracy
    print(f"{dataset_name} Accuracy Per Label:")
    for label, acc in per_label_accuracy.items():
        print(f"  {label}: {acc:.2f}%")
        
    # Print per-label F1 score
    print(f"{dataset_name} F1 Score Per Label:")
    for label, f1 in per_label_f1.items():
        print(f"  {label}: {f1:.3f}")
        
    # Print total F1 score
    print(f"{dataset_name} Total F1 Score: {total_f1_score:.2f}%")

# Display final results
print("\nFinal Evaluation Results:")
display_results(model, train_loader, "Training")
display_results(model, val_loader, "Validation")
display_results(model, test_loader, "Test")