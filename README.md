#  Twitter Sentiment Analysis - Natural Language Processing

## 📌 Overview

This project performs **sentiment analysis** on tweets, with a focus on detecting **hate speech** using Natural Language Processing (NLP) and supervised machine learning models. The analysis involves extensive **text preprocessing**, **feature extraction**, and **model training and evaluation** to classify tweets as hate speech or not.

---

## 📁 Dataset

- **Source**: [Kaggle - Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets)
- **Size**: 31,962 labeled tweets
- **Labels**:
  - `0` = Hate Speech
  - `1` = Offensive Language
  - `2` = Neither

---

## 🧰 Tools & Technologies

### 🔤 Text Preprocessing
- `NLTK`: Tokenization, stopword removal
- `TextBlob`: Word stemming
- `TF-IDF Vectorizer` and `Count Vectorizer`: Feature extraction

### 🧠 Machine Learning
- `Scikit-learn`: Model building and evaluation
- `XGBoost`, `LightGBM`: Gradient boosting classifiers
- `Logistic Regression`: Baseline model

### 📊 Visualization
- `Matplotlib`: Performance charts and insights

### 📚 Data Manipulation
- `Pandas`, `NumPy`

---

## 🧪 Project Workflow

1. **Data Loading**
   - Import dataset from Kaggle
   - Explore class balance and structure

2. **Text Preprocessing**
   - Lowercasing, punctuation removal
   - Tokenization and stopword removal using NLTK
   - Stemming using TextBlob
   - Feature engineering with CountVectorizer and TF-IDF

3. **Model Building**
   - Models trained:
     - Logistic Regression
     - LightGBM
     - XGBoost
   - Evaluated using ROC-AUC, accuracy, and confusion matrix

4. **Model Evaluation**
   - Selected best model (XGBoost) based on performance metrics

---

## 📈 Model Performance Summary

| Model               | Accuracy | ROC-AUC Score |
|--------------------|----------|---------------|
| Logistic Regression | 0.83     | 0.85          |
| LightGBM            | 0.86     | 0.88          |
| XGBoost             | 0.88     | 0.89 ✅        |

> ✅ **XGBoost** performed the best overall and was selected for final deployment/testing.

---

## 📊 Dashboard & Visualizations

The project includes the following visual outputs:
- Confusion matrix
- Feature importance plot
- ROC curves
- Class distribution bar chart

---
Here’s an improved and **cleaned-up** version of the **Project Structure** and **How to Run** sections with:

* Better formatting
* Clearer descriptions
* Corrected code block syntax
* Removed unnecessary backslashes in filenames

---

## 🗂️ Project Structure

```text
twitter-sentiment-analysis/
├── data/                         # Contains the original dataset
│   └── tweets.csv
├── notebooks/                    # Step-by-step Jupyter Notebooks
│   ├── 01_data_exploration.ipynb       # Data loading and initial analysis
│   ├── 02_text_preprocessing.ipynb     # Tokenization, stopword removal, stemming
│   ├── 03_feature_engineering.ipynb    # Vectorization using TF-IDF & CountVectorizer
│   ├── 04_model_training.ipynb         # ML model implementation and training
│   └── 05_evaluation.ipynb             # Model performance evaluation
├── models/                      # Trained and saved models
│   └── final_model_xgboost.pkl
├── images/                      # Visualization outputs (ROC curves, confusion matrix, etc.)
│   └── confusion_matrix.png
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License file
└── README.md                    # Project documentation
```

---

##  How to Run

Follow the steps below to set up and run the project locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebooks:**

   Open the `.ipynb` files in the `notebooks/` folder using **Jupyter Notebook** or **JupyterLab**, and run them in the following order:

   1. `01_data_exploration.ipynb`
   2. `02_text_preprocessing.ipynb`
   3. `03_feature_engineering.ipynb`
   4. `04_model_training.ipynb`
   5. `05_evaluation.ipynb`

> ✅ Ensure Python 3.7+ is installed and compatible versions of required packages are used.

---


## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

---

---

##  Acknowledgments

* Kaggle for the dataset
* NLTK, TextBlob, Scikit-learn, LightGBM, and XGBoost for their powerful libraries
* Open-source contributors and the data science community

```


