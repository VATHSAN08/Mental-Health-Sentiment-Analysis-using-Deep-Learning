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

A robust preprocessing pipeline was implemented to clean and normalize text data, preparing it for effective model training. Key steps include:

* **Contraction Expansion**: Expanding contractions (e.g., "don't" to "do not").
* **Special Token Replacement**: Replacing URLs (`¡url¿`), user mentions (`¡user¿`), and hashtags (`¡hashtag¿`) with generic tokens.
* **Acronym Expansion**: Expanding common acronyms (e.g., "lol" to "laugh out loud," "mh" to "mental health").
* **Digit Handling**: Replacing specific digit-words (e.g., "gr8" to "great") and removing digits within other words.
* **Repeated Character Reduction**: Reducing sequences of 3 or more identical characters to two (e.g., "oooooh" to "ooh").
* **Punctuation and Special Character Removal**: Removing non-alphanumeric characters while preserving our custom tokens.
* **Stopword Removal**: Eliminating common English stopwords (though noted that Transformer models can handle them).
* **Lemmatization**: Reducing words to their base forms using spaCy (with NLTK as a fallback).
* **Text Filtering**: Removing texts shorter than a specified minimum word count (e.g., 5 words) after cleaning.
    * *Post-filtering Sample Sizes*: 35,957 training texts, 9,004 test texts.
* **Label Distribution (Cleaned Training Data)**:
    * Depression: 33.77%, Suicidal: 23.20%, Normal: 20.78%, Anxiety: 7.97%, Bipolar: 6.16%, Stress: 5.74%, Personality disorder: 2.39%.

### 📊 Exploratory Data Analysis (EDA)

Comprehensive EDA was performed on the cleaned training data to understand sentiment distribution and key textual patterns. Visualizations generated include:

* **Sentiment Distribution Bar and Pie Charts**: Visualizing the frequency and percentage breakdown of each sentiment class.
* **Word Clouds**: Generated for each sentiment class to highlight prominent words.
* **N-gram (Unigrams, Bigrams, Trigrams) Plots**: Displaying the most frequent word sequences for each sentiment.

**_Please insert your EDA plots here (e.g., sentiment distribution bar/pie charts, word clouds for each sentiment, N-gram visuals)._**

### 🧪 Models Implemented

This project explores both traditional machine learning and advanced deep learning approaches for sentiment classification.

1.  **Logistic Regression (Baseline Model)**
    * **Vectorization**: Utilized TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features (`max_features=5000`, `ngram_range=(1, 2)` for unigrams and bigrams).
    * **Hyperparameter Tuning**: Performed using `GridSearchCV` with `f1_weighted` scoring, exploring various `solver` (`liblinear`, `saga`), `penalty` (`l1`, `l2`), `C` values, and `class_weight` settings.

2.  **RoBERTa (Transformer Fine-Tuned)**
    * **Model**: Employed `roberta-base` for sequence classification (`RobertaForSequenceClassification`).
    * **Tokenizer**: Used `RobertaTokenizerFast` for efficient text tokenization and formatting.
    * **Customization**: Configured with a custom `dropout rate: 0.2` for regularization (`hidden_dropout_prob`, `attention_probs_dropout_prob`) and a `target weight decay: 0.01`.
    * **Loss Function**: Utilized `nn.CrossEntropyLoss` with dynamically computed `balanced` class weights (e.g., `[1.79, 2.32, 0.42, 0.69, 5.98, 2.49, 0.62]`) to effectively address class imbalance during training.
    * **Optimizer**: `AdamW` with a specified learning rate and weight decay.
    * **Learning Rate Scheduler**: Implemented `get_linear_schedule_with_warmup` (10% warmup steps out of 14,161 total training steps) for optimized learning rate adjustment.
    * **Training Strategy**: Trained for 7 epochs, incorporating validation monitoring and early stopping (with `PATIENCE_EPOCHS`) to prevent overfitting. The model achieved a **best validation accuracy of 0.7386**.

---

## 📈 Results

| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Logistic Regression    | 71.00%   | 0.71     |
| RoBERTa (fine-tuned)   | 75.33%   | 0.75     |

---

### Confusion Matrix plot for the RoBERTa model

Confusion matrices provide a detailed view of model performance, showing correct and incorrect classifications for each sentiment class.

**_Please insert your Confusion Matrix plot for the RoBERTa model here (e.g., `roberta-base_confusion_matrix_test_reg.png`)._**

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

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT). You are free to modify and distribute the code, provided you acknowledge the original source.

---

## 📧 Contact

For any inquiries or collaborations, feel free to reach out. 🤝

---

This notebook is a compassionate application of deep learning to:
* Classify mental health-related sentiments.
* Amplify unseen emotional voices.
* Enable proactive mental health support.

Join us as we explore the data with sensitivity, build powerful models with purpose, and unveil the emotional heartbeat of mental health text. Together, we can make a difference. 🤝✨
```
