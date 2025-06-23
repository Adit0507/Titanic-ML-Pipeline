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