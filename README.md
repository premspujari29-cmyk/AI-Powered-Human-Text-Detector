# AI vs Human Text Detection System 🤖📝

## 📌 Project Overview

This project is a Machine Learning and Natural Language Processing (NLP) based web application designed to classify text as either **AI-generated** or **human-written**. The system analyzes linguistic patterns, writing styles, and textual features to determine the origin of the input text with high accuracy.

The application provides a user-friendly interface where users can enter text manually or through voice input and receive real-time predictions along with confidence scores.

---

## 🚀 Features

* 🤖 Detects whether a given text is AI-generated or human-written
* 🧠 Uses Natural Language Processing (NLP) techniques for text analysis
* 📊 Provides prediction confidence scores for better interpretability
* 🎤 Supports voice-to-text input for hands-free text analysis
* ⚡ Real-time predictions through an interactive web interface
* 📈 Visualizes model performance and classification results
* 🌐 Easy-to-use Gradio-based frontend

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Gradio
* Speech Recognition

---

## 📊 Workflow

### 1️⃣ Input Collection

Users provide text either by:

* Typing directly into the application
* Using voice input through the microphone

### 2️⃣ Text Preprocessing

The input text undergoes:

* Lowercasing
* Removal of special characters and punctuation
* Tokenization and cleaning
* Feature preparation

### 3️⃣ Feature Extraction

The processed text is converted into numerical representations using:

* TF-IDF (Term Frequency-Inverse Document Frequency)

### 4️⃣ Model Prediction

The extracted features are passed to trained machine learning models such as:

* Random Forest
* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)

### 5️⃣ Classification Result

The system predicts whether the text is:

* ✅ Human Written
* 🤖 AI Generated

### 6️⃣ Result Visualization

The application displays:

* Prediction label
* Confidence score
* Classification insights

---

## 🎯 Applications

* Academic Integrity Verification
* AI Content Detection
* Content Authenticity Analysis
* Educational Research
* Content Moderation Systems
* Journalism and Publishing

---


```

---

## 🔮 Future Enhancements

* Deep Learning-based Detection using LSTM/BERT
* Multi-language Support
* Detection of AI Content from Different LLMs
* Improved Explainability and Model Interpretability
* Mobile Application Integration
* Cloud Deployment and API Support

---

## 📜 Conclusion

The AI vs Human Text Detection System demonstrates the practical application of Machine Learning and NLP in identifying AI-generated content. By combining robust text preprocessing, feature extraction, and classification techniques, the system provides an effective solution for content authenticity verification in the age of Generative AI.
