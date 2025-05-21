# 💖 Mental Health Sentiment Analysis using Deep Learning (RoBERTa)

## Project Overview

In a world where digital communication is increasingly prevalent, understanding and addressing mental well-being is paramount. This project, stands as a testament to our commitment to fostering a more empathetic and supportive online environment. It's not just about algorithms; it's about leveraging the power of AI to listen, understand, and ultimately, contribute to a kinder digital space.

At its heart, this initiative compassionately applies deep learning to classify mental health-related sentiments from textual data. By utilizing the robust RoBERTa model, this notebook meticulously analyzes textual expressions to identify various mental health sentiments such as Anxiety, Depression, Stress, Suicidal ideation, Bipolar disorder, and Personality disorder, alongside a 'Normal' category for general sentiments. Our goal is to provide a tool that can help in early detection and understanding, paving the way for timely support and intervention, and reminding us that behind every screen, there's a human story.


![Mental Health Banner](https://i.pinimg.com/736x/14/57/8e/14578edd117e0e6e99aebe86175953f9.jpg)

## 🌟 Project Overview

This project aims to classify mental health-related text into **seven distinct sentiment categories** — `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality Disorder`, `Stress`, and `Suicidal` — using both traditional and transformer-based deep learning models. It highlights the power of NLP in understanding unspoken emotions and potentially identifying early signs of mental health challenges.

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
- **Size**: 50,000+ labeled mental health text entries
- **Classes**: 7 multi-class categories

---

## 🔧 Methodology

### ✅ Data Preprocessing
- Contraction expansion
- Lowercasing, punctuation & stopword removal
- Lemmatization using spaCy
- Handling class imbalance with stratification

### 📊 Exploratory Data Analysis
- Word clouds
- Sentiment distribution plots
- Keyword analysis across mental health classes

### 🧪 Models Implemented
1. **Logistic Regression** (Baseline)  
   - Accuracy: **71%**  
2. **RoBERTa (Transformer Fine-Tuned)**  
   - Accuracy: **75.33%**, F1-Score: **0.75**

### ⚙️ Fine-Tuning Details
- Applied **weighted cross-entropy**, **dropout**, **L2 weight decay**, and a **linear warmup learning rate scheduler**
- Trained using PyTorch and HuggingFace Transformers

---

## 📈 Results

| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Logistic Regression    | 71.00%   | 0.71     |
| RoBERTa (fine-tuned)   | 75.33%   | 0.75     |

---

## 🚀 Installation & Running


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
