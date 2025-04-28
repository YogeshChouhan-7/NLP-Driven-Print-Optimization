import pandas as pd
import shutil
import os

# Define category priorities
category_priority = {
    'Medical': 1, 'Legal': 2, 'Finance': 3, 'Business': 4, 'Technological': 5,
    'Scientific': 6, 'News articles': 7, 'Government': 8, 'Educational': 9, 'Creative': 10
}

# Paths
csv_file_path = r'D:\projectfilehandling\venv\result.csv'
input_folder_path = r'D:\projectfilehandling\inputdata'
destination_folder_path = r'D:\projectfilehandling\destinationfolder'

# Step 1: Ensure destination folder exists
os.makedirs(destination_folder_path, exist_ok=True)

# Step 2: Read the CSV file
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    print(f"Failed to read CSV file: {e}")
    exit()

# Step 3: Add Priority column and sort by priority
df['Priority'] = df['Category'].map(category_priority)
df = df.dropna(subset=['Priority']).sort_values(by='Priority')

# Step 4: Move PDFs according to priority
for _, row in df.iterrows():
    # Prepend the input folder path to the file name
    pdf_path = os.path.join(input_folder_path, row['PDF Path'])

    if not os.path.isfile(pdf_path):
        print(f"File not found, skipping: {pdf_path}")
        continue

    # Move the PDF directly into the destination folder
    dest_pdf_path = os.path.join(destination_folder_path, os.path.basename(pdf_path))
    try:
        shutil.move(pdf_path, dest_pdf_path)
        print(f"Moved: {os.path.basename(pdf_path)} (Category: {row['Category']}) to {destination_folder_path}.")
    except Exception as e:
        print(f"Failed to move {pdf_path}: {e}")

print("\nAll available PDFs moved to destination folder based on priority!")
