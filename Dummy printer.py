import time
import os
import sys
import pandas as pd

# Priority order
category_priority = {
    'Medical': 1,
    'Legal': 2,
    'Finance': 3,
    'Business': 4,
    'Technological': 5,
    'Scientific': 6,
    'News articles': 7,
    'Government': 8,
    'Educational': 9,
    'Creative': 10
}

destination_folder_path = r'D:\projectfilehandling\destinationfolder'

# Read CSV
def load_pdf_paths():
    try:
        df = pd.read_csv(r'D:\projectfilehandling\venv\result.csv')
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

# Simulate printing a PDF
def print_pdf(file_name, category, duration=2):
    print(f"\nPrinting: {file_name} (Category: {category})")
    increments = 20
    bar_length = 30

    for i in range(increments + 1):
        time.sleep(duration / increments)
        progress = int((i / increments) * 100)
        filled_length = int(bar_length * i // increments)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {progress}%", end='')

    print(f"\nFinished printing: {file_name}")

    log_print_success(file_name, category)

# Log successful prints
def log_print_success(file_name, category):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("dummy_print_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] Printed: {file_name} (Category: {category})\n")

# Process and print all PDFs
def process_printing():
    df = load_pdf_paths()

    df['Priority'] = df['Category'].map(category_priority)
    df = df.dropna(subset=['Priority']).sort_values(by='Priority')

    for _, row in df.iterrows():
        pdf_filename = row['PDF Path']
        category = row['Category']
        full_pdf_path = os.path.join(destination_folder_path, pdf_filename)

        if not os.path.isfile(full_pdf_path):
            print(f"File not found, skipping: {pdf_filename}")
            continue

        print_pdf(pdf_filename, category)

# Main function
def main():
    print(f"Printer started. Monitoring {destination_folder_path}...\n")
    process_printing()
    print("\nAll files printed successfully!")

if __name__ == "__main__":
    if not os.path.exists(destination_folder_path):
        print(f"Destination folder does not exist: {destination_folder_path}")
        sys.exit(1)
    main()
