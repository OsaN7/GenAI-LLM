# # to read and manipulate the data
# import pandas as pd
# import numpy as np
# pd.set_option('max_colwidth', None)    # setting column to the maximum column width as per the data

# # to visualise data
# import matplotlib.pyplot as plt
# import seaborn as sns

# # to use regular expressions for manipulating text data
# import re

# # to load the natural language toolkit
# import nltk

# # to remove common stop words
# from nltk.corpus import stopwords

# # to perform stemming
# from nltk.stem.porter import PorterStemmer

# # to create Bag of Words (traditional approach)
# from sklearn.feature_extraction.text import CountVectorizer

# # to import Word2Vec for advanced embeddings
# from gensim.models import Word2Vec
# from gensim.utils import simple_preprocess

# # to split data into train and test sets
# from sklearn.model_selection import train_test_split

# # to build machine learning models
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# # to compute metrics to evaluate the model
# from sklearn import metrics
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # To tune different models
# from sklearn.model_selection import GridSearchCV

# # For text similarity and clustering
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# # For time series analysis
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')
# # Download required NLTK data
# import nltk

# # Download stopwords
# nltk.download('stopwords', quiet=True)

# # Download other required NLTK data for text preprocessing
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

# # Initialize the Porter Stemmer
# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()

# print("=== WORD2VEC ANALYSIS FOR NEWS ARTICLES ===")
# print("Loading Articles.csv dataset...")

# # Load the articles dataset
# try:
#     df = pd.read_csv('Articles.csv')
#     print(f"Dataset loaded successfully! Shape: {df.shape}")
#     print("\nColumns:", df.columns.tolist())
# except FileNotFoundError:
#     print("Articles.csv not found. Please ensure the file is in the correct directory.")
#     exit()

# # Display basic information about the dataset
# print("\n=== DATASET OVERVIEW ===")
# print("First few rows:")
# print(df.head(2))

# print(f"\nDataset shape: {df.shape}")
# print(f"Missing values:\n{df.isnull().sum()}")

# # For this analysis, we'll work with the 'Article text' column
# text_column = 'Article text'
# if text_column not in df.columns:
#     print(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
#     # Use the longest text column available
#     text_column = df.select_dtypes(include=['object']).columns[0]
#     print(f"Using column: {text_column}")

# print(f"\nWorking with column: '{text_column}'")
# print(f"Sample text length: {len(str(df[text_column].iloc[0]))}")

# # Create a copy for processing
# data = df.copy()

# print("\n=== TEXT PREPROCESSING ===")

# def remove_special_characters(text):
#     """Remove special characters and keep only alphanumeric characters"""
#     if pd.isna(text):
#         return ""
#     # Defining the regex pattern to match non-alphanumeric characters
#     pattern = '[^A-Za-z0-9]+'
#     # Finding the specified pattern and replacing non-alphanumeric characters with a space
#     new_text = ''.join(re.sub(pattern, ' ', str(text)))
#     return new_text

# def preprocess_text(text):
#     """Complete text preprocessing pipeline"""
#     if pd.isna(text):
#         return []
    
#     # Convert to string and lowercase
#     text = str(text).lower()
    
#     # Remove special characters
#     text = remove_special_characters(text)
    
#     # Remove extra whitespaces
#     text = ' '.join(text.split())
    
#     # Remove stopwords and tokenize
#     stop_words = set(stopwords.words('english'))
#     words = text.split()
    
#     # Filter out stopwords and short words
#     filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
#     # Apply stemming
#     stemmed_words = [ps.stem(word) for word in filtered_words]
    
#     return stemmed_words

# # Apply preprocessing
# print("Preprocessing text data...")
# data['cleaned_text'] = data[text_column].apply(remove_special_characters)
# data['processed_tokens'] = data[text_column].apply(preprocess_text)

# # Remove empty documents
# data = data[data['processed_tokens'].apply(len) > 0].reset_index(drop=True)

# print(f"Number of documents after preprocessing: {len(data)}")
# print("Sample processed tokens:", data['processed_tokens'].iloc[0][:10])

# print("\n=== WORD2VEC MODEL TRAINING ===")

# # Prepare data for Word2Vec
# sentences = data['processed_tokens'].tolist()

# # Train Word2Vec model
# print("Training Word2Vec model...")
# w2v_model = Word2Vec(
#     sentences=sentences,
#     vector_size=100,      # Dimensionality of word vectors
#     window=5,             # Context window size
#     min_count=2,          # Ignore words with frequency less than this
#     workers=4,            # Number of worker threads
#     epochs=10,            # Number of training epochs
#     sg=0                  # 0 for CBOW, 1 for Skip-gram
# )

# print(f"Word2Vec model trained successfully!")
# print(f"Vocabulary size: {len(w2v_model.wv.key_to_index)}")

# # Save the model
# model_path = 'articles_word2vec.model'
# w2v_model.save(model_path)
# print(f"Model saved as: {model_path}")

# print("\n=== WORD2VEC MODEL ANALYSIS ===")

# # Function to get document vector by averaging word vectors
# def get_document_vector(tokens, model, vector_size=100):
#     """Get document vector by averaging word vectors"""
#     vectors = []
#     for token in tokens:
#         if token in model.wv.key_to_index:
#             vectors.append(model.wv[token])
    
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros(vector_size)

# # Create document vectors
# print("Creating document vectors...")
# doc_vectors = []
# for tokens in data['processed_tokens']:
#     vector = get_document_vector(tokens, w2v_model)
#     doc_vectors.append(vector)

# doc_vectors = np.array(doc_vectors)
# print(f"Document vectors shape: {doc_vectors.shape}")

# print("\n=== WORD SIMILARITY ANALYSIS ===")

# # Find most common words
# all_words = []
# for tokens in sentences:
#     all_words.extend(tokens)

# word_freq = pd.Series(all_words).value_counts()
# print("Top 10 most frequent words:")
# print(word_freq.head(10))

# # Test word similarities
# test_words = ['econom', 'technolog', 'russia', 'china', 'covid']
# available_test_words = [word for word in test_words if word in w2v_model.wv.key_to_index]

# print(f"\nWord similarity analysis for available words: {available_test_words}")

# for word in available_test_words[:3]:  # Test first 3 available words
#     try:
#         similar_words = w2v_model.wv.most_similar(word, topn=5)
#         print(f"\nWords similar to '{word}':")
#         for sim_word, similarity in similar_words:
#             print(f"  {sim_word}: {similarity:.3f}")
#     except Exception as e:
#         print(f"Error finding similarities for '{word}': {e}")

# print("\n=== DOCUMENT CLUSTERING ANALYSIS ===")

# # Perform K-means clustering on document vectors
# n_clusters = 5
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# clusters = kmeans.fit_predict(doc_vectors)

# # Add cluster labels to dataframe
# data['cluster'] = clusters

# print(f"Documents clustered into {n_clusters} groups:")
# for i in range(n_clusters):
#     cluster_size = sum(clusters == i)
#     print(f"Cluster {i}: {cluster_size} documents")

# # Show sample headlines from each cluster
# if 'Headline' in data.columns:
#     print("\nSample headlines from each cluster:")
#     for i in range(n_clusters):
#         cluster_docs = data[data['cluster'] == i]
#         if len(cluster_docs) > 0:
#             print(f"\nCluster {i} samples:")
#             for j, headline in enumerate(cluster_docs['Headline'].head(3)):
#                 print(f"  {j+1}. {headline[:100]}...")

# print("\n=== TOPIC ANALYSIS BY CATEGORY ===")

# if 'Category' in data.columns:
#     print("Distribution by Category:")
#     category_dist = data['Category'].value_counts()
#     print(category_dist)
    
#     # Analyze clusters by category
#     print("\nCluster distribution by Category:")
#     cluster_category = pd.crosstab(data['cluster'], data['Category'])
#     print(cluster_category)

# print("\n=== DIMENSIONALITY REDUCTION VISUALIZATION ===")

# # PCA for visualization
# pca = PCA(n_components=2, random_state=42)
# doc_vectors_2d = pca.fit_transform(doc_vectors)

# # Create visualization
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], 
#                      c=clusters, cmap='tab10', alpha=0.6)
# plt.colorbar(scatter)
# plt.title('Document Clusters Visualization (Word2Vec + PCA)')
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('articles_word2vec_clusters.png', dpi=300, bbox_inches='tight')
# plt.show()

# print("\n=== DOCUMENT SIMILARITY ANALYSIS ===")

# # Find most similar documents
# def find_similar_documents(doc_index, doc_vectors, data, top_n=3):
#     """Find most similar documents to a given document"""
#     target_vector = doc_vectors[doc_index].reshape(1, -1)
#     similarities = cosine_similarity(target_vector, doc_vectors)[0]
    
#     # Get indices of most similar documents (excluding the document itself)
#     similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
#     return similar_indices, similarities[similar_indices]

# # Test similarity for first document
# if len(data) > 5:
#     print("Finding similar documents for the first article...")
#     similar_indices, similarities = find_similar_documents(0, doc_vectors, data, top_n=3)
    
#     print(f"\nOriginal article headline: {data.iloc[0]['Headline'] if 'Headline' in data.columns else 'N/A'}")
#     print("Most similar articles:")
    
#     for i, (idx, sim) in enumerate(zip(similar_indices, similarities)):
#         headline = data.iloc[idx]['Headline'] if 'Headline' in data.columns else f"Document {idx}"
#         print(f"{i+1}. Similarity: {sim:.3f}")
#         print(f"   Headline: {headline}")

# print("\n=== TEMPORAL ANALYSIS ===")

# if 'Date published' in data.columns:
#     # Convert date column to datetime
#     data['date'] = pd.to_datetime(data['Date published'], errors='coerce')
    
#     # Group by month and analyze
#     monthly_counts = data.groupby(data['date'].dt.to_period('M')).size()
    
#     print("Articles per month:")
#     print(monthly_counts.head(10))
    
#     # Plot temporal distribution
#     plt.figure(figsize=(12, 6))
#     monthly_counts.plot(kind='line', marker='o')
#     plt.title('Number of Articles Over Time')
#     plt.xlabel('Month')
#     plt.ylabel('Number of Articles')
#     plt.xticks(rotation=45)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('articles_temporal_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()

# print("\n=== WORD2VEC MODEL SUMMARY ===")
# print(f"✓ Processed {len(data)} articles")
# print(f"✓ Vocabulary size: {len(w2v_model.wv.key_to_index)} words")
# print(f"✓ Vector dimensionality: {w2v_model.wv.vector_size}")
# print(f"✓ Documents clustered into {n_clusters} groups")
# print(f"✓ Model saved as: {model_path}")
# print(f"✓ Visualizations saved as PNG files")

# print("\n=== USAGE EXAMPLES ===")
# print("To use the trained Word2Vec model:")
# print("1. Load model: model = Word2Vec.load('articles_word2vec.model')")
# print("2. Get word vector: model.wv['word']")
# print("3. Find similar words: model.wv.most_similar('word')")
# print("4. Calculate word similarity: model.wv.similarity('word1', 'word2')")

# print("\nAnalysis complete! 🎉")

