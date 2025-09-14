import pandas as pd
import re
from nltk.corpus import stopwords

def clean_text(text):
    """Clean text: lowercase, remove punctuation, numbers, stopwords."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_and_clean_data(fake_path, true_path, output_filename='fake_or_real_news_cleaned.csv'):
    """
    Load, label, clean, and combine fake/real news datasets from Google Drive.
    Saves cleaned CSV and returns DataFrame.
    """
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add label (0 = fake, 1 = real)
    fake_df['label'] = 0
    true_df['label'] = 1

    # Keep only title + label
    fake_df = fake_df[['title', 'label']].copy()
    true_df = true_df[['title', 'label']].copy()

    # Combine + shuffle
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop missing values
    combined_df.dropna(subset=['title'], inplace=True)

    # Drop duplicates
    combined_df.drop_duplicates(subset=['title'], inplace=True)

    # Clean titles
    combined_df['clean_title'] = combined_df['title'].apply(clean_text)

    # Save cleaned file
    combined_df.to_csv(output_filename, index=False)

    print("\n✅ SUCCESS! File saved as:", output_filename)
    print(f"Total samples: {len(combined_df)}")
    print(f"Fake: {len(combined_df[combined_df['label'] == 0])}")
    print(f"Real: {len(combined_df[combined_df['label'] == 1])}")

    return combined_df
