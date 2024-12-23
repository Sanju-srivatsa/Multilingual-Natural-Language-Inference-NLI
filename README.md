# Multilingual Natural Language Inference (NLI) Project

---
## Kaggle Notebook: 
https://www.kaggle.com/code/sanjusrivatsa9/multilingual-natural-language-inference-nli


## **Introduction**

Natural Language Inference (NLI), sometimes called Recognizing Textual Entailment (RTE), is an essential task in the field of Natural Language Processing (NLP). At its core, NLI involves **determining the relationship between two sentences** and understanding how they relate to each other. These two sentences are called:

1. **Premise**: This is the "starting point" or the first sentence. It presents a statement or fact that we accept as true.
   - Example: *"All dogs are animals."*
2. **Hypothesis**: This is the second sentence, which we analyze to determine how it relates to the premise.
   - Example: *"Some animals are dogs."*

### **Types of Relationships in NLI**
NLI classifies the relationship between the premise and the hypothesis into one of three categories:
1. **Entailment**: The hypothesis logically follows from the premise.  
   - Example:  
     Premise: *"All dogs are animals."*  
     Hypothesis: *"Some animals are dogs."*  
     **Relationship**: Entailment (The hypothesis is true if the premise is true.)
2. **Contradiction**: The hypothesis directly contradicts the premise.  
   - Example:  
     Premise: *"All dogs are animals."*  
     Hypothesis: *"No animals are dogs."*  
     **Relationship**: Contradiction (The hypothesis is false if the premise is true.)
3. **Neutral**: The hypothesis neither follows from nor contradicts the premise.  
   - Example:  
     Premise: *"All dogs are animals."*  
     Hypothesis: *"Some animals live in the ocean."*  
     **Relationship**: Neutral (The hypothesis is unrelated to the premise.)

### **Why is NLI Important?**
Understanding relationships between sentences is crucial in many real-world applications:
- **Customer Support**: Automatically understanding user queries and providing logical responses.
- **Legal Contracts**: Identifying contradictory clauses in agreements.
- **Search Engines**: Improving query results by understanding the intent behind user queries.
- **Content Moderation**: Detecting contradictions in claims, such as fake news detection.

---

## **Real-World Example of NLI**
Imagine a virtual assistant like Amazon Alexa or Siri:
- **Scenario**: A user asks, *"Can I park here for free?"*  
- The assistant retrieves a statement (premise): *"Parking is free for the first two hours."*
- **Analysis**: The assistant uses NLI to determine if the user's question (hypothesis) aligns with the premise.  
- **Result**:  
  - If the user plans to park for 1 hour: *Entailment* (Yes, it is free.)
  - If the user plans to park for 5 hours: *Contradiction* (No, it is not free after 2 hours.)
  - If the user asks about parking for bicycles: *Neutral* (The statement does not mention bicycles.)

---

## **Objective of the Project**
This project focuses on **multilingual NLI**, which adds an additional layer of complexity: **handling multiple languages**. For instance, the premise may be in English, but the hypothesis could be in French, Spanish, or any other language. This requires the model to:
1. Understand linguistic nuances across languages.
2. Accurately identify relationships despite variations in sentence structure and word usage.

---

## **Key Challenges in Multilingual NLI**
1. **Language Diversity**: Words and sentence structures differ significantly across languages.
   - Example: In English, we say *"The car is red."* while in Spanish, it is *"El coche es rojo."*
2. **Semantic Nuances**: A word might have multiple meanings based on context.
   - Example: The word *"bank"* could mean a financial institution or the side of a river.
3. **Data Representation**: Multilingual data often requires advanced models to represent text effectively across languages.

---

## **Real-World Applications**
1. **Multilingual Chatbots**:
   - Chatbots like those used in e-commerce must handle customer queries in multiple languages and provide consistent responses.
2. **Content Moderation**:
   - Platforms like Facebook and YouTube can use multilingual NLI to detect misinformation or contradictory statements in user posts.
3. **Legal Analysis**:
   - Comparing clauses in contracts written in different languages to identify inconsistencies or contradictions.
4. **Translation Verification**:
   - Ensuring that translated text retains the original meaning.

**Example**:  
Imagine a travel website where a user asks in German, *"Ist das Frühstück inklusive?"* (Is breakfast included?).  
The system retrieves a statement in English: *"The hotel offers a complimentary breakfast for all guests."*  
Using NLI, the system can infer the relationship and respond with *Entailment*, confirming that breakfast is included.

---

## **Why Multilingual NLI Matters?**
In today’s globalized world, businesses and platforms interact with users across different languages. By enabling systems to understand and infer relationships across multiple languages, **multilingual NLI bridges the gap between linguistic and cultural barriers**.

This project demonstrates the implementation of advanced multilingual NLP models to tackle these challenges, enabling systems to handle multilingual scenarios effectively.

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
![image](https://github.com/user-attachments/assets/566afce4-9436-434b-a7dd-949652d628cf)

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
