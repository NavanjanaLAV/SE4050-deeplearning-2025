## SE4050 Deep Learning Assignment: Fake News Detection from Headlines 

### 🔍 Problem Statement  
We developed a deep learning system to detect **fake news from headlines** — a critical challenge in today’s misinformation landscape. Using real-world social media-style headlines, our models classify text as “real” or “fake” based solely on linguistic patterns, without relying on external context or full articles.

### 📊 Dataset  
We used the **Fake and Real News Headlines Dataset**, originally sourced from Kaggle:  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  

This dataset was provided as two separate files:  
- `fake.csv` — 2,564 fake news headlines  
- `true.csv` — 2,563 real news headlines

We combined and cleaned these into a unified, standardized dataset:  
📄 **`Dataset/fake_or_real_news_cleaned.csv`**  
- Total samples: **5,127**  
- Columns: `title` (cleaned headline text), `label` (`real` or `fake`)  
- Class distribution: Balanced (50% fake, 50% real)  
- Preprocessing: Lowercase, removed URLs, punctuation, numbers, and empty entries

> 💡 All four models were trained and evaluated on this **exact same dataset** to ensure fair comparison.
