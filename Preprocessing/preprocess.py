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
    Load, label, clean, and combine fake/real news datasets.
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
    data = pd.concat([fake_df, true_df], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean titles
    data['title'] = data['title'].astype(str).apply(clean_text)

    # Save only title + label
    data.to_csv(output_filename, index=False)
    print(f"Cleaned dataset saved to {output_filename}")

    return data

if __name__ == "__main__":
    load_and_clean_data("Fake.csv", "True.csv")
