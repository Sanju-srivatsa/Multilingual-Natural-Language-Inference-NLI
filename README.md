# Multilingual Natural Language Inference (NLI) Project

---
## Kaggle Notebook: 
https://www.kaggle.com/code/sanjusrivatsa9/multilingual-natural-language-inference-nli

## **Introduction**

Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), is a foundational task in Natural Language Processing (NLP). The goal of NLI is to determine the semantic relationship between two sentences: a **premise** and a **hypothesis**. These relationships are classified into three categories:

1. **Entailment**: The hypothesis logically follows from the premise.
2. **Contradiction**: The hypothesis contradicts the premise.
3. **Neutral**: The relationship between the premise and hypothesis is unclear or unrelated.

This project tackles the NLI task in a **multilingual context**, where the models need to process and analyze text in multiple languages, accounting for the unique semantic, syntactic, and linguistic variations across languages. 

The focus is on comparing the performance of two advanced NLP models for multilingual NLI:
- **XLM-RoBERTa**: A transformer-based multilingual model known for its robust performance on multilingual tasks.
- **Sentence Transformers**: A lightweight embedding-based model designed for semantic similarity and related tasks.

The project aims to evaluate the effectiveness of these models using a multilingual dataset and assess their trade-offs in terms of accuracy, efficiency, and computational requirements.

---

## **Project Workflow**

This project follows a structured workflow comprising the following phases:

1. **Data Exploration and Preprocessing**
2. **Model Selection**
3. **Model Training and Evaluation**
4. **Result Analysis and Visualization**
5. **Comparison of Models**
6. **Conclusion and Insights**

---

## **1. Data Exploration and Preprocessing**

### **Dataset Overview**
The dataset consists of **premise-hypothesis pairs** across multiple languages, with the following key columns:
- **Premise**: The original sentence.
- **Hypothesis**: The sentence to be evaluated for entailment with the premise.
- **Label**: The target classification (0: Neutral, 1: Entailment, 2: Contradiction).
- **Language Information**: Language-specific details including language abbreviations (e.g., `en` for English).

### **Exploratory Data Analysis (EDA)**
EDA was performed to gain insights into the dataset, including:
- **Class Distribution**: Examining the frequency of labels to ensure balanced representation.
- **Language Diversity**: Analyzing the distribution of samples across languages.
- **Text Lengths**: Evaluating the lengths of premise and hypothesis sentences to understand data complexity.

#### **Key Observations**
- The dataset is diverse, with 15 different languages.
- Labels are nearly balanced with slight variations across the dataset.
- Sentence lengths vary significantly, affecting model tokenization and processing.

### **Preprocessing Steps**
1. **Missing and Duplicate Data Handling**:
   - Checked for and removed any missing values or duplicate entries to ensure data quality.
2. **Tokenization**:
   - Premises and hypotheses were concatenated with a separator token (`[SEP]`) and tokenized using model-specific tokenizers.
3. **Dataset Splitting**:
   - The dataset was split into **training (80%)**, **validation (20%)**, and **test** subsets for model training and evaluation.

---

## **2. Model Selection**

Two models were evaluated for multilingual NLI:

### **XLM-RoBERTa**
- **Type**: Transformer-based model.
- **Strengths**:
  - Robust performance on multilingual tasks.
  - Captures deep semantic relationships using self-attention mechanisms.
- **Fine-Tuning**:
  - Fine-tuned on the NLI dataset using Hugging Face's `Trainer` class.

### **Sentence Transformers**
- **Type**: Lightweight embedding-based model.
- **Strengths**:
  - Computationally efficient.
  - Effective for semantic similarity and text-pair classification tasks.
- **Approach**:
  - Embeddings for premises and hypotheses were computed, and cosine similarity was used to determine relationships.

---

## **3. Model Training and Evaluation**

### **Training Setup**
- **XLM-RoBERTa**:
  - Fine-tuned using Hugging Face Transformers.
  - Training included techniques such as mixed precision and dynamic padding.
  - Metrics: Accuracy, Precision, Recall, and F1 Score.

- **Sentence Transformers**:
  - Pre-trained embeddings were computed for premises and hypotheses.
  - Predictions were made using cosine similarity and a threshold-based classification.

### **Evaluation Metrics**
1. **Accuracy**: Proportion of correctly predicted labels.
2. **Precision**: Ratio of true positives to all predicted positives.
3. **Recall**: Ratio of true positives to all actual positives.
4. **F1 Score**: Harmonic mean of precision and recall.

---

## **4. Results**

### **XLM-RoBERTa Performance**
- **Accuracy**: 89%
- **Precision**: 90%
- **Recall**: 89%
- **F1 Score**: 90%
- **Insights**: The model performed exceptionally well, capturing complex semantic relationships across languages.

### **Sentence Transformers Performance**
- **Accuracy**: 26%
- **Precision**: 39%
- **Recall**: 26%
- **F1 Score**: 26%
- **Insights**: While computationally efficient, the model struggled with nuanced semantic relationships, limiting its effectiveness for NLI.

---

## **5. Visualization and Comparison**

### **Performance Comparison**
A comparative bar chart was created to visualize the performance metrics of the two models. Key findings:
- **XLM-RoBERTa**: Consistently outperformed Sentence Transformers in all metrics.
- **Sentence Transformers**: Highlighted the trade-off between computational efficiency and accuracy.

---

## **6. Conclusion and Insights**

- **XLM-RoBERTa** emerged as the superior model for multilingual NLI tasks, offering robust accuracy and a deep understanding of semantic relationships.
- **Sentence Transformers**, while lightweight and efficient, lacked the complexity to handle nuanced relationships, making it less ideal for NLI.

### **Applications and Use Cases**
- **XLM-RoBERTa** is recommended for applications requiring high accuracy and handling complex multilingual datasets.
- **Sentence Transformers** may be suitable for lightweight, real-time applications where efficiency is prioritized over accuracy.

### **Future Work**
- Experiment with other multilingual models (e.g., mT5 or multilingual BERT).
- Investigate hybrid approaches combining transformer models with lightweight embeddings.
- Extend the analysis to domain-specific NLI tasks.

---

## **How to Run the Project**

### **1. Prerequisites**
- Python 3.8+
- Libraries: `transformers`, `datasets`, `sentence-transformers`, `sklearn`, `matplotlib`, `pandas`, `seaborn`.

### **2. Running the Code**
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the notebook to explore the dataset, train models, and evaluate results.

### **3. Project Files**
- `train.csv` and `test.csv`: Multilingual NLI datasets.
- `submission.csv`: Predicted labels for the test set.
- Notebook: Includes all implementation steps.

---

## **Acknowledgments**
- **Kaggle**: Dataset and competition platform.
- **Hugging Face**: Transformer-based NLP models and tools.
- **Sentence Transformers**: Efficient semantic similarity models.

This project serves as a comprehensive exploration of multilingual NLI and showcases the trade-offs between different modeling approaches, making it a valuable addition to your portfolio.
