            # unsupervised and transformer
# !pip install -q transformers datasets
# !pip install -U sentence-transformers -q
# pip install -U sentence-transformers
# pip install torch
# pip install scikit-learn torch
import pandas as pd
import numpy as np
pd.set_option('max_colwidth',None)
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy.spatial.distance import cdist,pdist
from sklearn.metrics import silhouette_score
import torch    
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')   
from google.colab import drive
drive.mount('/C:\Users\rajbh\OneDrive\Desktop\LLMnGenAI\News_Article')
data=review.copy()
data.loc[1,'Text']
data.head()
data.tail()
data.shape
