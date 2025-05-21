# 💖 Mental Health Sentiment Analysis using Deep Learning (RoBERTa)

## 🌟 Project Overview

In our digital world, prioritizing mental well-being is critical. This project, uses AI to compassionately listen, understand, and foster a kinder online space. By analyzing sentiments, we provide vital early insights and support, highlighting how crucial this analysis is for accessible mental healthcare. This project, stands as a testament to our commitment to fostering a more empathetic and supportive online environment.

At its heart, this initiative compassionately applies deep learning to classify mental health-related sentiments from textual data.  
This project aims to classify mental health-related text into **seven distinct sentiment categories** — `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality Disorder`, `Stress`, and `Suicidal` — using both traditional and transformer-based deep learning models. It highlights the power of NLP in understanding unspoken emotions and potentially identifying early signs of mental health challenges. 
Our goal is to provide a tool that can help in early detection and understanding, paving the way for timely support and intervention, and reminding us that behind every screen, there's a human story.

<p align="center">
  <img src="https://i.pinimg.com/736x/14/57/8e/14578edd117e0e6e99aebe86175953f9.jpg" alt="Mental Health Banner" width="300"/>
</p>

> Behind every message is a human story — and these tools can help make those voices heard.

---

## 🧠 Motivation

In the digital era, individuals often express their deepest struggles through online platforms. Sentiment analysis offers a way to detect and interpret these emotional cues, providing:

- 🆘 **Early intervention** for at-risk individuals  
- 🌐 **Public mental health insights** to shape policies  
- ❤️ **Stigma reduction** through empathetic AI  
- 🧠 **Tailored support systems** via classification-driven responses  

---

## 🔍 Dataset

- **Source**: [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- **Size**: Over 50,000 labeled mental health text entries.
- **Classes**: 7 multi-class categories: `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality disorder`, `Stress`, `Suicidal`.
- **Initial Data Split**:
    - Full Training Set Size: 42,434 samples
    - Raw Test Set Size: 10,609 samples

---

## 🔧 Methodology

### ✅ Data Preprocessing

A robust preprocessing pipeline cleaned and normalized text data, including:

* **Text Cleaning**: Expanding contractions, replacing special tokens (URLs, mentions, hashtags), expanding acronyms, handling digits, reducing repeated characters, and removing punctuation.
* **Linguistic Processing**: Removing stopwords and lemmatizing words (using spaCy/NLTK).
* **Filtering**: Removing short texts (less than 5 words after cleaning).
* **Label Distribution**: Analysis of sentiment class distribution (Depression: 33.77%, Suicidal: 23.20%, Normal: 20.78%, etc.).

### 📊 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed to understand sentiment distribution and textual patterns. Visualizations generated include:

* **Sentiment Distribution Charts**: Bar and Pie charts.
* **Word Clouds**: Per sentiment class.
* **N-gram Plots**: Unigrams, Bigrams, and Trigrams per sentiment.
Here is the Sentiment Distribution plot:

<p align="center">
  <img src="https://github.com/indranil143/Mental-Health-Sentiment-Analysis-using-Deep-Learning/blob/main/SS/Sentiment%20Distribution.png" alt="Sentiment Distribution" width="600"/>
</p>

### 🧪 Models Implemented

This project utilized both traditional machine learning and advanced deep learning models:

1.  **Logistic Regression (Baseline Model)**
    * **Vectorization**: TF-IDF with `max_features=5000` and `ngram_range=(1, 2)`.
    * **Hyperparameter Tuning**: `GridSearchCV` with `f1_weighted` scoring across various solvers, penalties, and C values.

2.  **RoBERTa (Transformer Fine-Tuned)**
    * **Model**: Fine-tuned `roberta-base` for sequence classification.
    * **Configuration**: Used `RobertaTokenizerFast`, custom `dropout rate: 0.2`, and `weight decay: 0.01`.
    * **Training**: Employed `nn.CrossEntropyLoss` with balanced class weights, `AdamW` optimizer, and `linear warmup scheduler`. Trained for 7 epochs with validation monitoring and early stopping, achieving a **best validation accuracy of 0.7386**.

---

## 📈 Results

| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Logistic Regression    | 71.00%   | 0.71     |
| RoBERTa (fine-tuned)   | 75.33%   | 0.75     |

---

### Confusion Matrix plot for the RoBERTa model

<p align="center">
  <img src="https://github.com/indranil143/Mental-Health-Sentiment-Analysis-using-Deep-Learning/blob/main/SS/CM%20-%20RoBERTa.png" alt="RoBERTa Confusion Matrix" width="600"/>
</p>

---

## ✨ Prediction Result Example

The model can lovingly process raw text and output the predicted sentiment along with the probability distribution across all defined sentiment classes, offering insights that can guide compassionate responses.

**Example 1: Positive Sentiment**
* **Original Text:** "I am feeling absolutely ecstatic and overjoyed today, everything is wonderful!"
* **Cleaned Text:** "feel absolutely ecstatic overjoyed today everything wonderful"
* **Predicted Sentiment:** Normal
* **Class Probabilities:**
    * Anxiety: 0.0111, Bipolar: 0.0693, Depression: 0.1111, **Normal: 0.7580**, Personality disorder: 0.0073, Stress: 0.0021, Suicidal: 0.0412

**Example 2: Depression Sentiment**
* **Original Text:** "The weight of this sadness is crushing me. I feel so empty and depressed."
* **Cleaned Text:** "weight sadness crush feel empty depressed"
* **Predicted Sentiment:** Depression
* **Class Probabilities:**
    * Anxiety: 0.0011, Bipolar: 0.0012, **Depression: 0.9742**, Normal: 0.0023, Personality disorder: 0.0015, Stress: 0.0006, Suicidal: 0.0191

<p align="center">
  <img src="https://github.com/indranil143/Mental-Health-Sentiment-Analysis-using-Deep-Learning/blob/main/SS/prediction%20example.png" alt="Prediction Example" width="400"/>
</p>

---

## 🚀 Installation & Running

To get this project up and running, follow these simple steps:

# Clone the repository
```bash
git clone https://github.com/indranil143/Mental-Health-Sentiment-Analysis-using-Deep-Learning.git
cd Mental-Health-Sentiment-Analysis-using-Deep-Learning
```
# Install required libraries
```bash
pip install -r requirements.txt
```
# Run the notebook
```bash
jupyter notebook Mental_Health_Sentiment_Analysis_(RoBERTa).ipynb
```
---

## 🤝 Future Work and Contributions

This project is an open invitation to join us in making a positive impact. Contributions are warmly welcomed! If you're inspired to enhance this endeavor, consider:

* **Expanding the Dataset:** Let's collaboratively grow the dataset with more diverse and larger collections, enriching the model's understanding of the human experience.
* **Exploring Other Models:** Venture into new horizons by experimenting with other transformer models or deep learning architectures, constantly seeking better ways to connect and comprehend.
* **Deploying the Model:** Imagine an API or a compassionate web application for real-time sentiment analysis, bringing immediate understanding and support to those who need it most.
* **Bias Analysis:** With a gentle hand and keen eye, investigate and mitigate potential biases in the model's predictions, ensuring fairness and equity in our digital empathy.

---

## 📄 License

This project is open-source and available under the [MIT License](https://github.com/indranil143/Mental-Health-Sentiment-Analysis-using-Deep-Learning/blob/main/LICENSE). You are free to modify and distribute the code, provided you acknowledge the original source.

---

## 📧 Contact

For any inquiries or collaborations, feel free to reach out. 🤝

---

This notebook is a compassionate application of deep learning to:
* Classify mental health-related sentiments.
* Amplify unseen emotional voices.
* Enable proactive mental health support.

Join us as we explore the data with sensitivity, build powerful models with purpose, and unveil the emotional heartbeat of mental health text. Together, we can make a difference. 🤝✨
