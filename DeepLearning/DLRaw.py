"""
TreatmentFlow
Deep Learning: Patient Priority Prediction

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


    def format_patient_rows(self, patients_set):
        # lite preprocessing function made specifically for formatting each patient in hospital

        # # Remove rows with NaN values in the target variable
        # DataFrame = DataFrame.dropna(subset=['esi'])
        
        # # Convert 'esi' to int to ensure it's a proper class label
        # DataFrame['esi'] = DataFrame['esi'].astype(int)
        
        # # Continue with the rest of preprocessing
        # exclude_cols = ['dep_name', 'esi', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status']
        # feature_cols = [col for col in DataFrame.columns if col not in exclude_cols]
        # num_features = DataFrame[feature_cols].select_dtypes(include=[np.number])
        # cat_features = DataFrame[feature_cols].select_dtypes(include=['object', 'category', 'bool'])
        

        if 'esi' in patients_set.columns:
            patients_set = patients_set.drop(columns=['esi'])

        # Ensure column order matches training input
        expected_cols = self.df.drop(columns=['esi']).columns
        for col in expected_cols:
            if col not in patients_set.columns:
                patients_set[col] = 0  # or np.nan if preferred for imputation

        patients_set = patients_set[expected_cols]  # re-order columns

        # Transform using the existing fitted preprocessor
        processed_array = self.preprocessor.transform(patients_set)

        
        # Retrieve output column names from the preprocessor
        column_names = self.preprocessor.get_feature_names_out()

        # Wrap the processed array back into a DataFrame
        processed_df = pd.DataFrame(processed_array, columns=column_names)

        return processed_df


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
    
    def prep_dl(self):


        # TESTING FUNCTION, DELETE IN FINAL ******************************************

        # current_dir = os.path.dirname(__file__)
        # parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        # directory = os.path.join(parent_dir, 'CTAS_files')

        # # process data
        # DataFrame = self.load_data(directory)
        # x_train, x_test, y_train, y_test, preprocessor = self.preprocess_data(DataFrame)

        # # build model
        # model, start_time = self.build_model()

        # # train model
        # model = self.train_model(model, x_train, y_train)

        # deploy model
        sample_index = np.random.choice(range(len(self.df)), size=1, replace=False)
        sample_patient = self.df.drop('esi', axis=1).iloc[sample_index]
        result = self.predict_priority(self.model, sample_patient, self.preprocessor)

        actual_esi = self.df.iloc[sample_index]['esi'].values[0]

        print("\nSample Patient Predictions:")
        print(f"Prediction: {result[0]}, Actual: {actual_esi}")

    def predict_esi(self, patient_dict=None):
        # sample_index = np.random.choice(range(len(self.df)), size=1, replace=False)
        # sample_patient = self.df.drop('esi', axis=1).iloc[sample_index]
        # result = self.predict_priority(self.model, sample_patient, self.preprocessor)

        # ISSUE: NEW SAMPLE PATIENT USED AT EACH INVOCATION
        

        # Step 1: Sample one real patient row from the dataset (drop 'esi')
        sample_index = np.random.choice(range(len(self.df)), size=1, replace=False)
        sample_patient = self.df.drop(columns=['esi']).iloc[sample_index]

        # sample_patient = pd.DataFrame(sample_patient, columns=self.df.drop(columns=['esi']).columns)

        user_patient = sample_patient.copy()

        # UPDATE SYMPTOMS BASED ON TRIAGE
        # only update specific values if patient performs triage
        if isinstance(patient_dict, dict):
            # Step 2: Update sample patient with survey responses
            for key, value in patient_dict.items():
                if key in user_patient.columns:
                    # THIS DICT ACCESSING MAY NOT WORK WITH NP DF
                    user_patient.at[0, key] = value

        # format sample patient to fit model before changing
        # formatted_patient = self.format_patient_rows(sample_patient)


        # # patient_df = pd.DataFrame([sample_patient])
        # # Step 4: Match column set and order to training data
        # expected_cols = self.df.drop(columns=['esi']).columns
        # for col in expected_cols:
        #     if col not in sample_patient.columns:
        #         sample_patient[col] = 0  # set column to dummy value

        # fit df to correct column order & values
        # sample_patient = sample_patient[expected_cols]


        # Step 4: Predict priority
        prediction = self.predict_priority(self.model, user_patient, self.preprocessor)

        # Step 5: Output result
        print(f"\nPredicted ESI: {prediction[0]}")
        return user_patient, prediction[0]














# def main():
#     # Initialize the model
#     print("\nWelcome to the TreatmentFlow Deep Learning module!")
#     print("\n -------------------------------------------------------------------------")
#     print("\nLoading data...")
    
#     # get file directory from adjacent folder (same parent)

#     current_dir = os.path.dirname(__file__)
#     parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
#     directory = os.path.join(parent_dir, 'CTAS_files')

#     # directory = 'C:/Users/emese/Desktop/TreatmentFlow/CTAS_files'
#     DataFrame = load_data(directory)
#     print("\n -------------------------------------------------------------------------")
#     print("\nPreprocessing data...")
#     x_train, x_test, y_train, y_test, preprocessor = preprocess_data(DataFrame)
#     print("Data loading and preprocessing complete.")



#     # 5 levels of patient priority
#     priority_mapping = {
#             1: "Immediate (Resuscitation)",
#             2: "Emergency",
#             3: "Urgent",
#             4: "Less Urgent",
#             5: "Non-Urgent"
#         }
#     # Build the model
#     print("\n -------------------------------------------------------------------------")
#     print("\nBuilding deep learning model...")
#     model, start_time = build_model(x_train, x_test, y_train, y_test)
#     # Train the model
#     print("\n -------------------------------------------------------------------------")
#     print("\nTraining deep learning model...")
#     model = train_model(model, start_time, x_train, y_train)

#     # Evaluate the model    
#     print("\n -------------------------------------------------------------------------")
#     print("\nEvaluating deep learning model...")
#     model = evaluate_model(model, start_time, x_test, y_test)

#     # Predict patient priority for sample patients
#     print("\n -------------------------------------------------------------------------")
#     print("\nPredicting patient priority for sample patients.")
#     sample_indices = np.random.choice(range(len(DataFrame)), size=5, replace=False)
#     sample_patients = DataFrame.drop('esi', axis=1).iloc[sample_indices]
#     results = predict_priority(model, sample_patients, preprocessor, priority_mapping)

#     # Display results
#     print("\nSample Patient Predictions:")
#     print(results[['predicted_priority', 'priority_description']])
    
#     print("\nTreatmentFlow Deep Learning module completed successfully!")
    
#     # Feature importance analysis (if available with the model)
#     try:
#         feature_importances = model.feature_importances_
#         print("\nTop 10 Most Important Features:")
#         # This would need additional code to map feature indices back to names
#     except:
#         print("\nFeature importance not available for this model type.")

#     print("\nTreatmentFlow Deep Learning module completed successfully!")

if __name__ == "__main__":
    dl = DeepLearningTriage()

    dl.prep_dl()