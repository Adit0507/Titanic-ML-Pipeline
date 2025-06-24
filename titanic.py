import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

class TitanicMLPipeline:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, file_path='titanic.csv'):
        # loadimng the titanic dataset from csv
        try:
            if file_path.endswith('.csv'):
                try:
                    self.full_df = pd.read_csv(file_path)
                except:
                    self.full_df = pd.read_excel(file_path) #if csv fails, will try to read as an excel

            else:   #read as excel file
                self.full_df = pd.read_excel(file_path) 

            print("Data loaded successfully")
            print(f"Dataset shape: {self.full_df.shape}")
            print(f"Columns: {self.full_df.columns}")

            # checkin if this is a combined dataset or a separate test
            if 'Survived' in self.full_df.columns :
                self.train_df= self.full_df[self.full_df['Survived'].notna()].copy()
                self.test_df = self.full_df[self.full_df['Survived'].isna()].copy()

                if len(self.test_df) == 0:
                    print("Creating test/train split from full dataset")
                    from sklearn.model_selection import train_test_split
                    self.train_df, test_temp =train_test_split(self.full_df, test_size=0.2, random_state=42, stratify=self.full_df['Survived'])
                    self.test_df = test_temp.drop('Survived', axis=1)

            else :
                print("No 'Survived' colum found- treating as test data ")
                print("Generating sample training data for demo")
                
                self.test_df = self.full_df.copy()
                self.generate_sample_data()
                return
            
            print(f"Training set shape: {self.train_df.shape}")
            print(f"Test set shape: {self.test_df.shape}")

        except FileNotFoundError:
            print(f"File '{file_path}' not found. Generating sample data for demo")
            self.generate_sample_data()

        except Exception as e:
            print(f"Error loading data", {e})
            print("Generating sample data for demonstration...")
            self.generate_sample_data()

    def generate_sample_data(self):
        np.random.seed(42)
        n_samples =891 

        data = {    #generating data
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29, 14, n_samples),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.4, n_samples),
            'Fare': np.random.lognormal(3.2, 1.0, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        }

        age_missing =  np.random.choice(n_samples, int(0.2*n_samples), replace=False)
        data['Age'][age_missing] = np.nan

        self.train_df = pd.DataFrame(data)
        self.test_df = self.train_df.sample(n=200).drop('Survived', axis=1).reset_index(drop=True)

        print("Sample data generated for demo")

    def explore_data(self):
        print("\n" + "="*50)
        print("Exploratory Data Analysis")
        print("="*50)

        print("\nDataset Info: ")
        print(self.train_df.info())

        print("\nMissing Values")
        missing_data= self.train_df.isnull().sum()
        missing_percent = 100* missing_data/ len(self.train_df)
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] >0])

        print("\nSurvival Rate by Feature")
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        for feature in categorical_features:
            if feature in categorical_features:
                survival_feature = self.train_df.groupby(feature)['Survived'].mean()
                print(f"\n{feature}: ")
                print(survival_feature)

    # creatin new features and preprocessing existing ones
    def feature_engineering(self):
        print("\n" "="*50)
        print("feature engineering")
        print("="*50)

        def engineer_features(df):
            df = df.copy()
            
            # Create Title feature from Name
            if 'Name' in df.columns:
                df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
                df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
                df['Title'] = df['Title'].replace('Mlle', 'Miss')
                df['Title'] = df['Title'].replace('Ms', 'Miss')
                df['Title'] = df['Title'].replace('Mme', 'Mrs')
            else:
                # synthetic titles based on sex and age
                df['Title'] = 'Mr'
                df.loc[df['Sex'] == 'female', 'Title'] = 'Miss'
                if 'Age' in df.columns:
                    df.loc[(df['Sex'] == 'female') & (df['Age'] > 18), 'Title'] = 'Mrs'
            
            # family size feature
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            
            # age groups
            if 'Age' in df.columns:
                df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Middle_aged', 'Senior'])
            
            # fare groups
            df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
            
            return df
        
        self.train_df = engineer_features(self.train_df)
        self.test_df = engineer_features(self.test_df)
        
        print("New features created:")
        new_features = ['Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']
        for feature in new_features:
            if feature in self.train_df.columns:
                print(f"- {feature}")

    # handling missing values and encoding categorical variables
    def preprocess_data(self):
        print("\n" + "="*50)
        print("Data preprocessing")
        print("="*50)

        # handlin missing values
        # age: fill with median based on Title
        if 'Age' in self.train_df.columns and self.train_df['Age'].isnull().any() :
            for title in self.train_df['Title'].unique():
                age_median = self.train_df[self.train_df['Title'] == title]['Age'].median()
                self.train_df.loc[(self.train_df['Title'] == title) & (self.train_df['Age'].isnull()), 'Age'] = age_median

                if title in self.test_df['Title'].unique():
                    self.test_df.loc[(self.test_df['Title'] == title) & (self.test_df['Age'].isnull()), 'Age'] = age_median

        # fill with mode
        if 'Embarked' in self.train_df.columns:
            embarked_mode = self.train_df['Embarked'].mode()[0]
            self.train_df['Embarked'].fillna(embarked_mode, inplace=True)
            self.test_df['Embarked'].fillna(embarked_mode, inplace=True)
        
        # filling with median
        if self.test_df['Fare'].isnull().any():
            fare_median = self.train_df['Fare'].median()
            self.test_df['Fare'].fillna(fare_median, inplace=True)

        # featurs for modelling
        feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Title', 'FamilySize', 'IsAlone']
        
        # only availabe features
        available_features = [col for col in feature_columns if col in self.train_df.columns]
        
        # preparing training data
        X = self.train_df[available_features].copy()
        y = self.train_df['Survived']

        # test data
        X_test= self.test_df[available_features].copy()
        self.X_test = X_test.copy()

        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le 

        # handling remaining missing values
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        X_test= pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)

        # split training data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # scalin features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns)
        self.X_val_scaled = pd.DataFrame(self.X_val_scaled, columns=self.X_val.columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=X_test.columns)

        print("Preprocessing completed")
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Features: {list(self.X_train.columns)}")

    def train_models(self):
        print("\n" + "="*50)
        print("Model Training")
        print("="*50)

        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        } 

        cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_scores= {}
        
        print("Training models with cross validation...")
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM', 'K-Nearest-Neighbors']:
                X_train_use = self.X_train_scaled
            else:
                X_train_use = self.X_train

            # cross validation
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=cv, scoring="accuracy")
            model_scores[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }

            # training on full training set
            model.fit(X_train_use, self.y_train)
            self.models[name] = model

            print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        best_model_name= max(model_scores.keys(), key=lambda x:model_scores[x]['mean_score'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        return model_scores

    def hyperparameter_tuning(self):
        print("\n" + "="*50)
        print("Hyperparameter Tuning")
        print("="*50) 

        # parameter grids for top models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        } 

        tuned_models = {}

        for model_name, param_grid in param_grids.items():
            if model_name in self.models:
                print(f"\nTuning {model_name}...")

                # getitn base model
                if model_name == "Random Forest":
                    base_model = RandomForestClassifier(random_state=42)
                    X_use = self.X_train

                elif model_name =="Gradient Boosting":
                    base_model= GradientBoostingClassifier(random_state=42)
                    X_use = self.X_train

                else :  #logistic regression
                    base_model = LogisticRegression(random_state=42, max_iter=1000)
                    X_use = self.X_train_scaled

                # grid search
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, )
                grid_search.fit(X_use, self.y_train)

                tuned_models[model_name] = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # updatin models with tuned versions
        self.models.update(tuned_models)
        print("\nHyperparameter tuning completed!")
    
    def evaluate_models(self):
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)

        results = {}

        for name, model in self.models.items():
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                X_val_use = self.X_val_scaled
            else:
                X_val_use = self.X_val 

            # predictions
            y_pred = model.predict(X_val_use)
            y_pred_proba = model.predict_proba(X_val_use)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            roc_auc = roc_auc_score(self.y_val, y_pred_proba) if y_pred_proba is not None else None

            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred
            }

            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            if roc_auc:
                print(f"  ROC AUC: {roc_auc:.4f}")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best validation accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # detailed report ffor best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(self.y_val, results[best_model_name]['predictions']))
        
        return results
    
    def feature_importance(self):
        print("\n" + "="*50)
        print("Feature Importance")
        print("="*50)

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns

            # feature importance data frame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print(f"Feature Importance ({self.best_model_name}):")
            print(importance_df)

            return importance_df
        
        else:
            print(f"Feature importance not available for {self.best_model_name}")
            return None 
        
    def make_predictions(self):
        print("\n" + "="*50)
        print("Final Predictions")
        print("="*50) 

        if self.best_model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
            X_test_use = self.X_test_scaled
        else: 
            X_test_use = self.X_test

        # makin predictions
        predictions = self.best_model.predict(X_test_use)
        predictions_probabilities= self.best_model.predict_proba(X_test_use)[:, 1] if hasattr(self.best_model, 'predict_proba') else None 

        # submission dataframe
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'] if 'PassengerId' in self.test_df.columns else range(len(predictions)),
            'Survived': predictions
        })

        if predictions_probabilities is not None:
            submission['Survival_Probability']= predictions_probabilities

        print(f"Predictions made using {self.best_model_name}")
        print(f"Predicted survival rate: {predictions.mean():.3f}")
        print(f"\nFirst 10 predictions:")
        print(submission.head(10))

        return submission

    def run_pipeline(self, file_path="titanic.csv"):
        print("Titanic Survival Prediction: ")
        print("="*60)

        # load and explore data
        self.load_data(file_path)
        self.explore_data()

        # feat. eng and preprocessing
        self.feature_engineering()
        self.preprocess_data()

        # model training and evaluation
        self.train_models()
        self.hyperparameter_tuning()
        evaluation_results = self.evaluate_models()

        # feature analysis
        self.feature_importance()

        final_predictions = self.make_predictions()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

        return final_predictions, evaluation_results

if __name__ == "__main__":
    pipeline = TitanicMLPipeline()
    predictions, results = pipeline.run_pipeline('titanic.csv')
    
    predictions.to_csv('titanic_predictions.csv', index=False)
    print("\nPredictions saved to 'titanic_predictions.csv'")

