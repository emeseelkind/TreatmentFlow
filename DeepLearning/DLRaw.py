"""
TreatmentFlow
Deep Learning: Patient Priority Prediction (Designed for Project's Component Integration)

Automate assignment of patient priority (1-5) using Triage information 
Training based on 500k+ observation dataset of triage information associated with patient priority

By Adam Neto and Emese Elkind
Started: February 2025

CISC 352: Artificial Intelligence

DNN (Deep Neural Network) supervised classification model for patient triage and diagnosis prediction
"""

"""
Step 1: Import Libraries
sklearn and tensorflow libraries used for data preprocessing and deep learning model building
"""
import pandas as pd
import numpy as np
# Splitting Data into Training & Test Sets
from sklearn.model_selection import train_test_split
# Feature Scaling & Encoding Categorical Variables
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# For building the deep learning model
from sklearn.neural_network import MLPClassifier
# For timing the training process
import time
import os

class DeepLearningTriage:

    def __init__(self):

        self.df = None
        self.model = None
        self.preprocessor = None

        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        directory = os.path.join(parent_dir, 'CTAS_files')

        # process data
        self.df = self.load_data(directory)
        x_train, y_train, self.preprocessor = self.preprocess_data(self.df)

        # build model
        self.model, start_time = self.build_model()

        # train model and save to object
        self.model = self.train_model(self.model, x_train, y_train)

        print(f"\nTriage Model Built and Trained in {(time.time() - start_time):.2f} seconds")

    def load_data(self, directory):

        # build dataframe from CTAS database
        combined_df = pd.DataFrame()
        
        # Loop through all CSV files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                df = df.dropna(subset=['esi']) # drop rows with missing esi values
                
                if 'esi' in df.columns:
                    combined_df = pd.concat([combined_df, df])
                
        combined_df = combined_df.reset_index(drop=True)

        return combined_df

    def preprocess_data(self, DataFrame):
        
        # Remove rows with NaN values in the target variable
        DataFrame = DataFrame.dropna(subset=['esi'])
        
        # Convert 'esi' to int to ensure it's a proper class label
        DataFrame['esi'] = DataFrame['esi'].astype(int)
        
        # Continue with the rest of preprocessing
        exclude_cols = ['dep_name', 'esi', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status']
        feature_cols = [col for col in DataFrame.columns if col not in exclude_cols]
        num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
        cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features.columns),
                ('cat', categorical_transformer, cat_features.columns)
            ]
        )
        
        x = DataFrame[feature_cols]
        y = DataFrame['esi'].values
        
        # Check for any remaining NaN values
        if np.isnan(y).any():
            raise ValueError("Target variable 'esi' still contains NaN values after preprocessing")
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train = preprocessor.fit_transform(x_train)
        return x_train, y_train, preprocessor
     
    def build_model(self):
        """Build the deep learning model for priority prediction"""
        
        # Create an MLPClassifier (Multi-Layer Perceptron - Neural Network)
        start_time = time.time()

        # MLPClassifier mimics a deep neural network architecture
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Three hidden layers similar to the Keras model
            activation='relu',             # ReLU activation function
            solver='adam',                 # Adam optimizer
            alpha=0.0001,                  # L2 regularization parameter
            batch_size=32,                 # Mini-batch size
            learning_rate_init=0.001,      # Initial learning rate
            max_iter=100,                  # Maximum number of iterations
            early_stopping=True,           # Use early stopping
            validation_fraction=0.1,       # Use 10% of training data for validation
            n_iter_no_change=10,           # Number of iterations with no improvement to wait before early stopping
            random_state=42,               # Random seed for reproducibility
            verbose=True                   # Display progress during training
        )
        
        return model , start_time
  
    def train_model(self, model, x_train, y_train):
        """Train the deep learning model"""
        
        # Train the model
        model.fit(x_train, y_train)

        return model

    def predict_priority(self, model, sample_patient, preprocessor):
        """Predict patient priority using the trained model"""

        # PREPROCESSING AGAIN COULD BE UNNECESSARY
        new_data_processed = preprocessor.transform(sample_patient)
        
        # Get predictions
        prediction = model.predict(new_data_processed)

        return prediction
    
    def predict_esi(self, patient_dict=None):
        
        # sample one real patient row from the dataset (drop 'esi') to act as the base for the user
        sample_index = np.random.choice(range(len(self.df)), size=1, replace=False)
        sample_patient = self.df.drop(columns=['esi']).iloc[sample_index]
        sample_patient.index = [0] # reset index so the updating using dict works

        # only update specific values if patient performs triage
        if isinstance(patient_dict, dict):
            
            # FLAG - dictionary is not being returned with the proper updated values

            for key, value in patient_dict.items():
                if key in sample_patient.columns:
                    # THIS DICT ACCESSING MAY NOT WORK WITH NP DF
                    sample_patient.at[0, key] = value

        # predict priority
        prediction = self.predict_priority(self.model, sample_patient, self.preprocessor)

        # output result
        return sample_patient, prediction[0]
