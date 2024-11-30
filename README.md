# IMDB Movie Review Sentiment Analysis 🎬📊

## 🚀 Project Description
This project implements a comprehensive sentiment analysis classifier for IMDB movie reviews using advanced natural language processing and machine learning techniques. The goal is to automatically classify movie reviews as positive or negative based on their textual content.

## 📋 Project Structure
```
imdb-sentiment-analysis/
│
├── IMDB Dataset.csv           # Raw dataset
├── model.pkl                  # Trained machine learning model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
└── sentiment_analysis.ipynb   # Main Jupyter Notebook
```

## 🛠 Technologies and Libraries Used
- **Data Manipulation**: 
  - `pandas`: Data processing and manipulation
  - `numpy`: Numerical computing

- **Natural Language Processing**:
  - `nltk`: Natural Language Toolkit
    - Stopwords removal
    - Porter Stemming

- **Text Processing**:
  - `re` (Regular Expressions): Text cleaning
  - `scikit-learn`: 
    - TF-IDF Vectorization
    - Text feature extraction

- **Machine Learning**:
  - `scikit-learn` Classifiers:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Support Vector Machine (Linear Kernel)

## 🔍 Detailed Preprocessing Pipeline

### 1. Text Cleaning
```python
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-alphabetic characters
    text = re.sub(r'^a-zA-Z\s', '', text)
    
    # Convert to lowercase
    text = text.lower()
    return text
```

### 2. Text Preprocessing
```python
def preprocess_text(text):
    # Tokenization
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)
```

## 🤖 Machine Learning Models Comparison

### Model Training and Evaluation
```python
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model 1: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_predictions = lr_model.predict(X_test_tfidf)

# Model 2: Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)

# Model 3: Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
```

## 💾 Model Persistence
```python
# Saving trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)  # Saving best performing model

# Saving vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Loading model for future use
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

## 📊 Performance Metrics
Evaluated using classification report with metrics:
- Precision
- Recall
- F1-Score
- Support

## 🔮 Potential Improvements
- Experiment with deep learning models (LSTM, BERT)
- Implement cross-validation
- Try advanced feature extraction techniques
- Explore ensemble methods

## 📦 Installation and Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analysis.git

# Install required packages
pip install pandas numpy scikit-learn nltk

# Download NLTK resources
python -c "import nltk; nltk.download('stopwords')"
```


**Happy Sentiment Analysis! 🎉**
