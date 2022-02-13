# Major NLP Library for Python
# 1. NLTK
# 2. SpaCy
# 3. Scikit-Learn
# 4. TextBlob
"""
# 5. Stanford CoreNLP
# 6. Stanford Parser
# 7. Stanford Dependency Parser
# 8. Stanford NER
# 9. Stanford Coreference Resolution
# 10. Stanford Sentiment Analysis
# 11. Stanford Semantic Role Labeling
"""

# Scikit-Learn Approach to NLP
""""
# 1. Vectorization
# 2. Feature Selection
# 3. Dimensionality Reduction
# 4. Clustering
# 5. Model Selection
# 6. Model Evaluation
# 7. Model Tuning
# 8. Model Deployment
"""
# Powerful library for process and anlyze text, images and extract information from documents
# Built-in modules
# Feature Extraction (Information Extraction from Text and Images)
# Model training (Analyze based on categories and train specific models, can be supervised or unsupervised)
# Pipeline Building (extract feature around word, streamline the process into a stages)
# Performance Optimization
# Grid Search (Search for the best parameters)


# To use txt files as input, we need to convert them into...
"""
"""
# a dataframe
# The dataframe is a table with rows and columns
# The dataframe can be used to train a model
# The text files are loaded with categories as sub folder names

# Build a feature extraction transformer
# from sklearn.feature_extraction.text import <appropriate transformer>
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer

# Data load Object - Helps to load data from datasets
# Attributes:
# 1. Bunch - Contains field that can be accessed as dict keys or objects
# 2. Target Names - List of required categories
# 3. Data - Refers to and attribute in memory

# Using digits dataset
from sklearn.datasets import load_digits

# Create object of loaded dataset
digit_dataset = load_digits()

# Use built in DESCR function to get the description of the dataset
# You can view all info that describe dataset.
print(digit_dataset.DESCR)

print(type(digit_dataset))

print(digit_dataset.data)

# Target (Response) data
print(digit_dataset.target)

# Feature extraction
# Technique to convert content into numerical vectors to perform machine learning
# Bag of words - Counts the number of times a word appears in a document
