# 💳 ApexCredit - Credit Card Eligibility Assessment Portal

ApexCredit is a web application that helps users check their credit card eligibility based on their personal and financial details.

The application uses Machine Learning to predict whether an applicant is likely to be eligible for a credit card and displays the result instantly.

---

# 🚀 Features

- Credit card eligibility prediction
- Simple and responsive web interface
- Real-time prediction
- Form validation
- Machine Learning model integration
- Result with eligibility score and recommendation

---

# 🛠 Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

---

# 📁 Project Structure

# 📁 Project Structure

```text
CreditCardApprovalPrediction/
│
├── app.py                      # Main Flask application
├── train_model.py              # Train and evaluate ML models
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignored files
├── LICENSE                     # License information
├── generate_docs.py            # Documentation generator
│
├── dataset/                    # Dataset files
│   ├── CreditCardApproval_Selected_Features.csv
│   └── CreditCardApproval_Selected_Features.xlsx
│
├── models/                     # Saved machine learning models
│   ├── best_model.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_info.json
│
├── utils/                      # Helper modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── helper.py
│   ├── model_trainer.py
│   └── model_loader.py
│
├── templates/                  # HTML pages
│   ├── base.html
│   ├── home.html
│   ├── predict.html
│   └── result.html
│
└── static/                     # Static files
    ├── css/
    │   ├── style.css
    │   ├── home.css
    │   └── predict.css
    │
    ├── js/
    │   ├── home.js
    │   └── predict.js
    │
    └── images/
        ├── Target_Distribution.png
        ├── Accuracy_Comparison.png
        ├── Balanced_Accuracy_Comparison.png
        ├── F1_Comparison.png
        ├── Precision_Comparison.png
        ├── Recall_Comparison.png
        ├── ROC_AUC_Comparison.png
        ├── Logistic_ConfusionMatrix.png
        ├── DecisionTree_ConfusionMatrix.png
        ├── RandomForest_ConfusionMatrix.png
        ├── XGBoost_ConfusionMatrix.png
        └── plots/
```

---

# 📋 Requirements

Install the following before running the project:

- Python 3.10 or above
- Git
- VS Code (Recommended)

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd CreditCardApprovalPrediction
```

---

## 2️⃣ Create a Virtual Environment

Windows

```bash
python -m venv venv
```

Linux/macOS

```bash
python3 -m venv venv
```

---

## 3️⃣ Activate the Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

If activated successfully, you will see:

```text
(venv)
```

---

## 4️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Check the Dataset

Make sure the dataset is inside:

```text
dataset/
```

If the dataset is already included in the repository, you can skip this step.

---

## 6️⃣ Train the Model (Optional)

If the **models** folder already contains the trained model files, you can skip this step.

Otherwise run:

```bash
python train_model.py
```

---

## 7️⃣ Run the Project

```bash
python app.py
```

---

## 8️⃣ Open the Website

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

# 📊 Application Flow

```text
Home Page
      ↓
Prediction Form
      ↓
Enter Details
      ↓
Submit
      ↓
View Result
```

---

# ❗ Common Problems

### Packages not installed

```bash
pip install -r requirements.txt
```

---

### Dataset not found

Check that the dataset is inside:

```text
dataset/
```

---

### Model not found

Train the model:

```bash
python train_model.py
```

---

### Virtual Environment not activated

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

---

# 👥 Team Guidelines

- Pull the latest code before starting work.

```bash
git pull origin main
```

- After making changes:

```bash
git add .
git commit -m "Your message"
git push origin main
```

- Do not upload:
  - `venv/`
  - `__pycache__/`
  - `.env`

- If you install a new package, update:

```bash
pip freeze > requirements.txt
```


