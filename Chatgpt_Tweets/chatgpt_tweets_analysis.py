# ChatGPT Tweets Sentiment Analysis - Complete Workflow
# Working with file.csv dataset

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Machine learning libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Word2Vec for advanced analysis
try:
    from gensim.models import Word2Vec
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    GENSIM_AVAILABLE = True
except ImportError:
    print("Gensim not available. Install with: pip install gensim")
    GENSIM_AVAILABLE = False

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("=== CHATGPT TWEETS SENTIMENT ANALYSIS ===")
print("Loading file.csv dataset...")

# Step 1: Load and explore the data
try:
    df = pd.read_csv('file.csv')
    print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
    print("✓ Columns:", df.columns.tolist())
except FileNotFoundError:
    print("❌ file.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Step 2: Data exploration
print("\n=== DATASET OVERVIEW ===")
print("First few rows:")
print(df.head())

print(f"\nDataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Check data types
print(f"\nData types:\n{df.dtypes}")

# Analyze the labels
print(f"\nSentiment distribution:")
sentiment_counts = df['labels'].value_counts()
print(sentiment_counts)

# Calculate percentages
sentiment_percentages = df['labels'].value_counts(normalize=True) * 100
print(f"\nSentiment percentages:")
for label, percentage in sentiment_percentages.items():
    print(f"{label}: {percentage:.2f}%")

# Step 3: Data visualization
print("\n=== DATA VISUALIZATION ===")

plt.figure(figsize=(15, 10))

# Sentiment distribution
plt.subplot(2, 3, 1)
sentiment_counts.plot(kind='bar', color=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Sentiment percentages (pie chart)
plt.subplot(2, 3, 2)
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=['lightgreen', 'lightcoral', 'lightblue'])
plt.title('Sentiment Distribution (Percentages)')

# Tweet length analysis
df['tweet_length'] = df['tweets'].astype(str).str.len()

plt.subplot(2, 3, 3)
df.boxplot(column='tweet_length', by='labels', ax=plt.gca())
plt.title('Tweet Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Length (characters)')

# Tweet length distribution
plt.subplot(2, 3, 4)
plt.hist(df['tweet_length'], bins=50, alpha=0.7, color='skyblue')
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length (characters)')
plt.ylabel('Frequency')

# Word count analysis
df['word_count'] = df['tweets'].astype(str).str.split().str.len()

plt.subplot(2, 3, 5)
df.boxplot(column='word_count', by='labels', ax=plt.gca())
plt.title('Word Count by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Word Count')

# Word count distribution
plt.subplot(2, 3, 6)
plt.hist(df['word_count'], bins=50, alpha=0.7, color='lightgreen')
plt.title('Distribution of Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('chatgpt_tweets_eda.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 4: Text preprocessing
print("\n=== TEXT PREPROCESSING ===")

# Initialize preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def remove_special_characters(text):
    """Remove special characters and keep only alphanumeric characters"""
    if pd.isna(text):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters
    pattern = '[^A-Za-z0-9]+'
    new_text = re.sub(pattern, ' ', text)
    return new_text

def preprocess_text(text):
    """Complete text preprocessing pipeline"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters, URLs, mentions, hashtags
    text = remove_special_characters(text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Remove stopwords and tokenize
    words = text.split()
    
    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Apply stemming
    stemmed_words = [ps.stem(word) for word in filtered_words]
    
    return ' '.join(stemmed_words)

# Apply preprocessing
print("Preprocessing tweets...")
df['cleaned_tweets'] = df['tweets'].apply(preprocess_text)

# Remove empty tweets
df = df[df['cleaned_tweets'].str.len() > 0].reset_index(drop=True)

print(f"Dataset shape after preprocessing: {df.shape}")
print("\nSample original tweet:")
print(df['tweets'].iloc[0])
print("\nSample preprocessed tweet:")
print(df['cleaned_tweets'].iloc[0])

# Step 5: Feature extraction and model training
print("\n=== FEATURE EXTRACTION & MODEL TRAINING ===")

# Prepare data for modeling
X = df['cleaned_tweets']
y = df['labels']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Method 1: Bag of Words (CountVectorizer)
print("\n--- Method 1: Bag of Words ---")
cv = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Train Random Forest with Bag of Words
rf_bow = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bow.fit(X_train_cv, y_train)

# Predictions and evaluation
y_pred_bow = rf_bow.predict(X_test_cv)
accuracy_bow = accuracy_score(y_test, y_pred_bow)

print(f"Bag of Words + Random Forest Accuracy: {accuracy_bow:.4f}")
print("\nClassification Report (Bag of Words):")
print(classification_report(y_test, y_pred_bow))

# Method 2: TF-IDF
print("\n--- Method 2: TF-IDF ---")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models with TF-IDF
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

# Display results comparison
print("\n=== MODEL COMPARISON ===")
print("Method comparison:")
print(f"Bag of Words + Random Forest: {accuracy_bow:.4f}")
for name, accuracy in results.items():
    print(f"TF-IDF + {name}: {accuracy:.4f}")

# Best model analysis
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nBest performing model: TF-IDF + {best_model_name} ({best_accuracy:.4f})")

# Detailed analysis of best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_tfidf)

print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print("✓ Data exploration completed")
print("✓ Text preprocessing completed")
print("✓ Multiple models trained and compared")
print("✓ Visualizations saved as PNG files")
print(f"✓ Best model: {best_model_name} with {best_accuracy:.4f} accuracy")

# Optional: Word2Vec analysis (if gensim is available)
if GENSIM_AVAILABLE:
    print("\n=== WORD2VEC ANALYSIS (OPTIONAL) ===")
    print("Training Word2Vec model...")
    
    # Prepare sentences for Word2Vec
    sentences = [text.split() for text in df['cleaned_tweets'] if text.strip()]
    
    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)
    
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv.key_to_index)}")
    
    # Find similar words
    test_words = ['chatgpt', 'openai', 'good', 'bad']
    available_words = [word for word in test_words if word in w2v_model.wv.key_to_index]
    
    if available_words:
        print("\nWord similarities:")
        for word in available_words[:3]:
            try:
                similar = w2v_model.wv.most_similar(word, topn=5)
                print(f"\nWords similar to '{word}':")
                for sim_word, score in similar:
                    print(f"  {sim_word}: {score:.3f}")
            except:
                pass

print("\n🎉 Analysis complete! Check the generated visualizations and results above.")
