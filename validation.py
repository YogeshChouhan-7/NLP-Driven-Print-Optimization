import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the CSV files
original_df = pd.read_csv(r'D:\projectfilehandling\venv\orginal.csv')
result_df = pd.read_csv(r'D:\projectfilehandling\venv\result.csv')

# Display first few rows to understand the structure
original_df.head(), result_df.head()

# Step 1: Merge the two DataFrames on 'PDF Path'
merged_df = pd.merge(original_df, result_df, on='PDF Path')

# Step 2: Extract true and predicted labels
y_true = merged_df['orginal_Category']
y_pred = merged_df['Category']

# Step 3: Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, zero_division=0)

print("=== Evaluation Metrics ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n=== Confusion Matrix ===")
print(conf_matrix)

print("\n=== Classification Report ===")
print(class_report)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=y_true.unique(), 
            yticklabels=y_true.unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
