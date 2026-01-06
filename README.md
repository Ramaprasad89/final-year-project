
# Lung Cancer Prediction Model

This repository contains a Python code for predicting lung cancer based on various factors, including age, gender, smoking history, and exposure to harmful substances like radon and asbestos. The project utilizes machine learning techniques such as Random Forest and Logistic Regression to analyze and predict lung cancer risk.

## Project Overview

The objective of this project is to build a model that can predict the likelihood of a person having lung cancer based on their health data. The dataset includes various features, including personal history (age, gender), smoking history (pack years), and exposure to harmful substances.

### Key Features:
- **Data Preprocessing**: Handling missing data, encoding categorical variables, and scaling numerical values.
- **Model Training**: Using machine learning models like Random Forest and Logistic Regression.
- **Evaluation**: The model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Getting Started

1. **Clone the Repository**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/yourusername/lung-cancer-prediction.git
   ```

2. **Install Dependencies**  
   Make sure to install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
   The necessary libraries include:
   - Pandas
   - NumPy
   - Matplotlib
   - Seaborn
   - Scikit-learn

3. **Dataset**  
   Ensure that the dataset `lung_cancer_dataset.csv` is available in the working directory. You can replace it with your dataset if necessary.

4. **Running the Code**  
   To run the code, use the following command:
   ```bash
   python lung_cancer_prediction.py
   ```

## Code Overview

### Import Libraries and Load Data
We begin by importing the necessary libraries and loading the dataset:
```python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### Data Preprocessing
The data is preprocessed by mapping categorical values to numeric values and handling missing data. Here's an example of how the data is preprocessed:
```python
# Map categorical values to numeric values
df['radon_exposure'] = df['radon_exposure'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'None': 0, 'Moderate': 1, 'Heavy': 2})
```

### Model Training
We use a RandomForestClassifier to build the predictive model:
```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### Model Evaluation
The model's performance is evaluated using accuracy, precision, recall, F1-score, and AUC-ROC:
```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
