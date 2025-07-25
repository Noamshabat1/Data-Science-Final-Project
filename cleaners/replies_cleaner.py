import pandas as pd
import os


def clean_replies():
    # Load replies data
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_file = os.path.join(project_root, "data", "splitted", "musk_replies.csv")
    output_file = os.path.join(project_root, "data", "clean", "clean_musk_replies.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Fill missing values with 0 for any column with "Count" in the name
    count_columns = [col for col in df.columns if 'Count' in col]
    for col in count_columns:
        df[col] = df[col].fillna(0)
    print(f"Filled missing values in Count columns: {count_columns}")
    
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Remove rows without any text
    initial_count = len(df)
    df = df.dropna(subset=['text'])  # Remove NaN/null
    df = df[df['text'].str.strip() != '']  # Remove empty/whitespace
    df = df[df['text'].str.contains(r'[a-zA-Z0-9]', regex=True, na=False)]  # Remove non-alphanumeric
    empty_text_removed = initial_count - len(df)
    print(f"Removed {empty_text_removed} rows without valid text")
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} cleaned rows to {output_file}")


if __name__ == "__main__":
    clean_replies()
