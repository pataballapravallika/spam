# 📩 Spam Message Classifier using Naive Bayes

This project is a simple yet effective spam detection system using **Natural Language Processing (NLP)** and **Multinomial Naive Bayes**. It classifies text messages as either "spam" or "ham" (not spam).

---

## 🗂️ Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **File**: `spam.csv`
- **Columns Used**:
  - `class`: Label (spam/ham)
  - `message`: The text message content

---

## 🔧 Technologies Used

- Python 3.x  
- Libraries:
  - `pandas`, `numpy` – Data manipulation
  - `scikit-learn` – NLP & machine learning

---

## 🚀 How It Works

1. **Data Preprocessing**
   - Loaded the dataset and selected only `class` and `message` columns.
   - Converted messages into numerical feature vectors using `CountVectorizer`.

2. **Model Building**
   - Used `MultinomialNB` (Naive Bayes) classifier suitable for text classification.
   - Split data into training and testing sets (67% train / 33% test).

3. **Prediction**
   - Accepts a custom input message from the user.
   - Classifies it as `"spam"` or `"ham"` based on the trained model.

---

## 📄 Sample Code

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[["class", "message"]]

# Features and labels
x = np.array(data["message"])
y = np.array(data["class"])

# Vectorize the text
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Prediction
sample = input('Enter a message: ')
data = cv.transform([sample]).toarray()
print(clf.predict(data))
