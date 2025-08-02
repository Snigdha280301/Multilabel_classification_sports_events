# Multi-Label Classification of Sports Events with BERT

This project performs **multi-label classification** on sports commentary using **BERT-based models**. It also extracts **players, teams, sentiment**, and predicts the **associated sport** for each commentary snippet. The workflow includes **data preprocessing, label generation, class balancing, model training, evaluation, and inference**.

---

##  Features
-  Multi-label classification using **BERT** with **custom loss (label smoothing)**.
-  Automatic label generation for categories:
  - **Score-related**
  - **Assist/Playmaking**
  - **Foul/Penalty**
  - **Substitution/Injury**
  - **Defense Actions**
  - **Game Outcome**
-  **Sport detection** using contextual keywords (Soccer, Cricket, Basketball, Tennis, etc.).
-  **Entity extraction** (Players & Teams) using **spaCy**.
-  **Sentiment analysis** using **VADER**.
-  Handles **imbalanced classes** using adjusted class weights.
-  **Fine-tuning BERT** with:
  - `BCEWithLogitsLoss` + Label smoothing
  - Gradient accumulation
  - Mixed precision (`fp16`)
  - Early stopping and model checkpointing
-  Inference pipeline for **new text commentary**.

---
##  Requirements

- Install the dependencies from requirements.txt
- Additionally, download SpaCy model and NLTK resources:

en_core_web_sm

stopwords

punkt

vader_lexicon

---
## How It Works

## 1. Data Preprocessing
- Load `.txt` files from `train_data/`.
- Clean text (remove punctuation, stopwords).
- Apply weighted keyword matching for label generation.

## 2. Labeling
- Multi-label encoding:
  [score-related, assist/playmaking, foul/penalty, substitution/injury, defense actions, game outcome]
- Detect associated sport using contextual keywords.

## 3. Model Training
- **Pre-trained model:** `bert-base-cased`
- **Problem type:** Multi-Label Classification
- **Loss:** Custom Label Smoothing over BCEWithLogitsLoss
- **Optimizer:** AdamW
- **Hyperparameters:**
- epochs = 15
- batch_size = 8
- learning_rate = 1e-5
- max_seq_length = 128

## 4. Evaluation
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score (weighted)**
- Handles class imbalance with adjusted weights.

## 5. Inference
Extract:
- **Players & Teams** (using spaCy NER)
- **Sentiment** (using VADER)
- **Predicted labels with probabilities**
- **Associated Sport**

---

## Example Prediction

**Input Commentary:**  
 Roger Federer faces Nadal in a historic Australian Open final.
 
**Output:**  
- Players: Roger Federer, Nadal  
- Teams: None  
- Sentiment: Positive  
- Associated Sport: Tennis  
- Predicted Labels: `['score-related', 'game outcome']`  
- Label Probabilities:  
  - score-related: 0.91  
  - game outcome: 0.78  

---

## Future Enhancements
- Add **NER fine-tuning** for better player/team extraction.
- Integrate explainability using **SHAP/LIME**.
- Deploy as an **API** using **FastAPI** or **Flask**.
- Support **multilingual commentary**.



