# 🚢 Titanic Survival Prediction ML Pipeline

A comprehensive machine learning pipeline that predicts passenger survival on the RMS Titanic with **83.2% accuracy** using advanced feature engineering and model optimization techniques.

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline to predict Titanic passenger survival. The pipeline handles real-world data challenges including missing values, feature engineering, model comparison, and hyperparameter optimization.

### 🎯 Key Results
- **Best Model**: Support Vector Machine (SVM)
- **Validation Accuracy**: 83.2%
- **ROC-AUC Score**: 0.85
- **Models Tested**: 6 different algorithms with hyperparameter tuning

## 🛠️ Features

### Data Processing
- **Intelligent Missing Value Handling**: Age imputation based on passenger titles, mode/median filling
- **Advanced Feature Engineering**: Created 5 new predictive features
- **Robust Data Loading**: Handles both CSV and Excel formats with error handling
- **Automatic Train/Test Splitting**: Creates validation sets for proper model evaluation

### Feature Engineering
- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, etc.) from passenger names
- **Family Size**: Combined SibSp and Parch for family dynamics
- **Age Groups**: Categorical age binning for better pattern recognition
- **Fare Categories**: Quartile-based fare groupings
- **Solo Traveler Flag**: Binary feature for traveling alone

### Model Pipeline
- **6 Algorithm Comparison**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes
- **Cross-Validation**: 5-fold stratified cross-validation for reliable performance estimates
- **Hyperparameter Tuning**: GridSearchCV optimization for top-performing models
- **Feature Scaling**: StandardScaler for distance-based algorithms

## 📈 Model Performance

| Model | Cross-Val Accuracy | Final Accuracy | ROC-AUC |
|-------|-------------------|----------------|---------|
| **SVM** | **82.4%** | **83.2%** | **0.85** |
| Gradient Boosting | 81.7% → 84.0%* | 82.5% | 0.89 |
| Random Forest | 78.7% → 83.7%* | 82.5% | 0.88 |
| Logistic Regression | 79.8% | 78.3% | 0.84 |
| Naive Bayes | 79.6% | 79.7% | 0.83 |
| K-Nearest Neighbors | 72.9% | 38.5% | 0.50 |

*After hyperparameter tuning

## 🔍 Key Insights

### Survival Patterns Discovered
- **Gender Impact**: Female survival rate (74%) vs Male (19%)
- **Class Hierarchy**: 1st Class (65%) > 2nd Class (45%) > 3rd Class (24%)
- **Port of Embarkation**: Cherbourg (56%) > Queenstown (44%) > Southampton (33%)

### Feature Importance
The engineered features significantly improved model performance, with passenger class, gender, and fare being the strongest predictors of survival.

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Usage
```python
from titanic_pipeline import TitanicMLPipeline

# Initialize and run the complete pipeline
pipeline = TitanicMLPipeline()
predictions, results = pipeline.run_pipeline('titanic.csv')

# Predictions saved automatically to 'titanic_predictions.csv'
```

### Input Data Format
The pipeline expects a CSV file with the following columns:
- `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`
- `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

## 📁 Project Structure
```
titanic-pipeline/
├── titanic.py              # Main pipeline implementation
├── titanic.csv             # Input dataset
├── titanic_predictions.csv # Output predictions
└── README.md              # This file
```

## 🧠 Technical Implementation

### Class Architecture
The `TitanicMLPipeline` class provides a complete ML workflow:
- `load_data()`: Flexible data loading with error handling
- `explore_data()`: Automated exploratory data analysis
- `feature_engineering()`: Advanced feature creation
- `preprocess_data()`: Missing value handling and encoding
- `train_models()`: Multi-algorithm training with cross-validation
- `hyperparameter_tuning()`: GridSearch optimization
- `evaluate_models()`: Comprehensive model evaluation
- `make_predictions()`: Final predictions with probability scores

### Error Handling
- Graceful handling of missing files with sample data generation
- Robust missing value imputation strategies
- Flexible input format support (CSV/Excel)

## 📊 Output
```
Best performing model: SVM
Best validation accuracy: 0.8322
Predicted survival rate: 0.369

Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.89      0.87        88
           1       0.80      0.75      0.77        55
    accuracy                           0.83       143
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **End-to-end ML pipeline development**
- **Feature engineering and domain knowledge application**
- **Model selection and hyperparameter optimization**
- **Cross-validation and proper model evaluation**
- **Clean, modular code architecture**
- **Real-world data preprocessing challenges**

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements. Areas for enhancement:
- Additional feature engineering techniques
- Ensemble methods implementation
- Model interpretability (SHAP, LIME)
- Interactive visualizations
- Deep learning approaches


*Built with ❤️ for learning and demonstrating machine learning engineering skills*