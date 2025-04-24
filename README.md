# AI-Driven-Prognosis-of-Traumatic-Brain-Injury-
# AI-Driven Prognosis of Traumatic Brain Injury using Random Forest

This project leverages machine learning—specifically the Random Forest algorithm—to predict the prognosis of patients suffering from Traumatic Brain Injury (TBI). The model is trained using key clinical features: **Glasgow Coma Scale (GCS)**, **Glasgow Outcome Scale (GOS)**, and **Age**. This approach aims to provide clinicians with a data-driven tool to support critical decision-making during acute care.

## 🧠 Project Objective

To develop a predictive model that estimates patient outcomes after TBI, based on early clinical indicators. Accurate prognosis can assist healthcare providers in:
- Prioritizing treatment strategies
- Communicating risks with families
- Optimizing resource allocation

## ⚙️ Features Used

- **Glasgow Coma Scale (GCS):** Assesses the level of consciousness.
- **Glasgow Outcome Scale (GOS):** Measures the outcome after brain injury.
- **Age:** Critical factor influencing recovery trajectory.

## 🧪 Methodology

- **Model Used:** Random Forest Classifier
- **Data Preprocessing:**
  - Handling missing values
  - Normalizing input data
- **Model Evaluation:**
  - Accuracy, Precision, Recall, F1-Score
  - Cross-validation
  - Confusion matrix visualization

## 📊 Dataset

The dataset consists of anonymized patient records with the following attributes:
- `Age` (Numeric)
- `GCS` score (Numeric: 3–15)
- `GOS` score (Numeric: 1–5, used as label/classification target)

> *Note: Ensure compliance with ethical data use guidelines and anonymization protocols.*


## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed. Then, clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/ai-tbi-prognosis.git
cd ai-tbi-prognosis
pip install -r requirements.txt
