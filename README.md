# Twitter Sentiment Analysis - Natural Language Processing
This project performs sentiment analysis on a Twitter dataset focused on detecting hate speech. The analysis includes data preprocessing steps such as tokenization, stemming, and the removal of stopwords. Multiple machine learning models are trained to classify tweets, including Gradient Boosting and XGBoost. The project's key components include:
- Loading and preprocessing a dataset from Kaggle
- Text feature extraction using TF-IDF and Count Vectorizer
- Implementing models including LightGBM, XGBoost, and Logistic Regression
- Evaluating model performance using ROC-AUC scores

**Tools Used**:
- **Libraries**: Scikit-learn, LightGBM, XGBoost, NLTK, TextBlob, Pandas, NumPy, Matplotlib
- **Data Source**: Twitter Sentiment Analysis dataset from Kaggle
- **Text Preprocessing**: TF-IDF Vectorizer, Count Vectorizer, NLTK stopwords, word stemming via TextBlob
- **Machine Learning Models**: Logistic Regression, Gradient Boosting (LightGBM), XGBoost
