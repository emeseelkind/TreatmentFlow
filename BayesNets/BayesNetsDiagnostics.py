import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the specified file path
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded data
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """
    Explore the dataset and provide basic statistics
    
    Parameters:
    df (pandas.DataFrame): The dataset to explore
    
    Returns:
    dict: Dictionary containing basic statistics
    """
    # Basic information
    stats = {
        'num_rows': df.shape[0],
        'num_cols': df.shape[1],
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    # Print basic stats
    print(f"Dataset has {stats['num_rows']} rows and {stats['num_cols']} columns")
    print("\nColumn names:")
    for col in stats['columns']:
        print(f"- {col}")
    
    print("\nMissing values:")
    for col, count in stats['missing_values'].items():
        if count > 0:
            print(f"- {col}: {count} missing values ({count/df.shape[0]*100:.2f}%)")
    
    return stats

def preprocess_data(df, target_column, categorical_columns=None, numeric_columns=None):
    """
    Preprocess the data for Bayesian analysis
    
    Parameters:
    df (pandas.DataFrame): The dataset to preprocess
    target_column (str): The target column for prediction
    categorical_columns (list): List of categorical columns to one-hot encode
    numeric_columns (list): List of numeric columns to standardize
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler, processed_df)
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].dtype == np.number:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        else:
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    # Process categorical columns if provided
    if categorical_columns:
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns, drop_first=True)
    
    # Separate features and target
    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]
    
    # Standardize numeric features if provided
    scaler = None
    if numeric_columns:
        valid_numeric_cols = [col for col in numeric_columns if col in X.columns]
        if valid_numeric_cols:
            scaler = StandardScaler()
            X[valid_numeric_cols] = scaler.fit_transform(X[valid_numeric_cols])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, processed_df

def build_bayesian_model(X_train, y_train, feature_names):
    """
    Build a Bayesian logistic regression model
    
    Parameters:
    X_train (numpy.ndarray): Training features
    y_train (numpy.ndarray): Training target
    feature_names (list): List of feature names for interpretability
    
    Returns:
    tuple: (model, trace)
    """
    n_features = X_train.shape[1]
    
    with pm.Model() as model:
        # Priors for model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)  # Intercept
        betas = pm.Normal('betas', mu=0, sd=1, shape=n_features)  # Coefficients
        
        # Linear combination of predictors
        eta = alpha + pm.math.dot(X_train, betas)
        
        # Logistic transformation
        p = pm.Deterministic('p', 1 / (1 + pm.math.exp(-eta)))
        
        # Likelihood (assuming binary classification)
        likelihood = pm.Bernoulli('likelihood', p=p, observed=y_train)
        
        # Sample from the posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, cores=1)
        
    # Create a dictionary mapping coefficients to feature names
    summary = az.summary(trace, var_names=['alpha', 'betas'])
    coefficients = {}
    coefficients['intercept'] = summary.loc['alpha', 'mean']
    
    for i, feature in enumerate(feature_names):
        coefficients[feature] = summary.loc[f'betas[{i}]', 'mean']
    
    # Sort by absolute value to identify most important features
    sorted_coeffs = {k: v for k, v in sorted(coefficients.items(), 
                                             key=lambda item: abs(item[1]) if k != 'intercept' else 0, 
                                             reverse=True)}
    
    print("Model coefficients (sorted by importance):")
    for feature, coeff in sorted_coeffs.items():
        print(f"- {feature}: {coeff:.4f}")
    
    return model, trace, coefficients

def evaluate_model(model, trace, X_test, y_test):
    """
    Evaluate the Bayesian model on test data
    
    Parameters:
    model (pymc3.Model): The trained Bayesian model
    trace (arviz.InferenceData): The trace from model sampling
    X_test (numpy.ndarray): Test features
    y_test (numpy.ndarray): Test target
    
    Returns:
    dict: Performance metrics
    """
    with model:
        # Extract posterior samples for alpha and betas
        alpha_samples = trace.posterior['alpha'].values.flatten()
        betas_samples = trace.posterior['betas'].values.reshape(-1, X_test.shape[1])
        
        # Number of posterior samples
        n_samples = len(alpha_samples)
        
        # Initialize array to store predictions
        y_pred_proba = np.zeros((n_samples, len(y_test)))
        
        # Generate predictions for each posterior sample
        for i in range(n_samples):
            # Linear combination
            eta = alpha_samples[i] + np.dot(X_test, betas_samples[i])
            # Apply logistic function
            y_pred_proba[i] = 1 / (1 + np.exp(-eta))
        
        # Calculate mean predicted probabilities
        mean_proba = np.mean(y_pred_proba, axis=0)
        # Convert to binary predictions using 0.5 threshold
        y_pred = (mean_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Confusion matrix components
        true_pos = np.sum((y_pred == 1) & (y_test == 1))
        true_neg = np.sum((y_pred == 0) & (y_test == 0))
        false_pos = np.sum((y_pred == 1) & (y_test == 0))
        false_neg = np.sum((y_pred == 0) & (y_test == 1))
        
        # Calculate precision, recall, and F1 score
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positive': true_pos,
                'true_negative': true_neg,
                'false_positive': false_pos,
                'false_negative': false_neg
            }
        }
        
        print(f"Model Performance:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- Precision: {precision:.4f}")
        print(f"- Recall: {recall:.4f}")
        print(f"- F1 Score: {f1_score:.4f}")
        print("\nConfusion Matrix:")
        print(f"- True Positives: {true_pos}")
        print(f"- True Negatives: {true_neg}")
        print(f"- False Positives: {false_pos}")
        print(f"- False Negatives: {false_neg}")
        
        return metrics, mean_proba

def visualize_results(trace, coefficients, feature_names, X_test, y_test, y_pred_proba):
    """
    Visualize the results of the Bayesian analysis
    
    Parameters:
    trace (arviz.InferenceData): The trace from model sampling
    coefficients (dict): Dictionary of model coefficients
    feature_names (list): List of feature names
    X_test (numpy.ndarray): Test features
    y_test (numpy.ndarray): Test target
    y_pred_proba (numpy.ndarray): Predicted probabilities
    """
    # Plot coefficient distributions (forest plot)
    plt.figure(figsize=(12, 8))
    az.plot_forest(trace, var_names=['betas'], combined=True)
    plt.title('Coefficient Distributions')
    plt.ylabel('Features')
    plt.show()
    
    # Plot posterior distributions
    plt.figure(figsize=(12, 8))
    az.plot_trace(trace, var_names=['alpha', 'betas'])
    plt.title('Posterior Distributions')
    plt.tight_layout()
    plt.show()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Feature importance plot
    sorted_coeffs = {k: v for k, v in sorted(coefficients.items(), 
                                           key=lambda item: abs(item[1]) if k != 'intercept' else 0, 
                                           reverse=True) if k != 'intercept'}
    
    plt.figure(figsize=(10, 8))
    features = list(sorted_coeffs.keys())
    values = list(sorted_coeffs.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    plt.barh(features, [abs(v) for v in values], color=colors)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def generate_diagnostic_report(coefficients, patient_data, feature_names):
    """
    Generate a diagnostic report for a given patient
    
    Parameters:
    coefficients (dict): Model coefficients
    patient_data (numpy.ndarray): Patient's feature values
    feature_names (list): Names of the features
    
    Returns:
    dict: Diagnostic report with probability and contributing factors
    """
     if isinstance(patient_data, dict):
        patient_data = pd.Series(patient_data)

    # Ensure all features are in patient_data
    missing_features = [f for f in feature_names if f not in patient_data]
    if missing_features:
        raise ValueError(f"Missing features in patient data: {missing_features}")
    
    # Extract intercept
    intercept = coefficients.get('intercept', 0)
    
    # Compute weighted contributions of features
    contributions = {}
    linear_combination = intercept
    for feature in feature_names:
        value = patient_data[feature]
        weight = coefficients.get(feature, 0)
        contribution = value * weight
        contributions[feature] = contribution
        linear_combination += contribution
    
    # Compute predicted probability using logistic function
    probability = 1 / (1 + np.exp(-linear_combination))
    prediction = int(probability >= 0.5)
    
    print("\n--- Diagnostic Report ---")
    print(f"Intercept: {intercept:.4f}")
    for feature, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"- {feature}: contribution = {contrib:.4f} (value: {patient_data[feature]}, weight: {coefficients.get(feature, 0):.4f})")
    print(f"\nLinear combination (logit): {linear_combination:.4f}")
    print(f"Predicted probability: {probability:.4f}")
    print(f"Predicted class (threshold 0.5): {prediction}")
    
    return {
        'intercept': intercept,
        'linear_combination': linear_combination,
        'predicted_probability': probability,
        'predicted_class': prediction,
        'feature_contributions': contributions
    }
    
    return report

def print_diagnostic_report(report, disease_name):
    """
    Print a formatted diagnostic report
    
    Parameters:
    report (dict): Diagnostic report from generate_diagnostic_report
    disease_name (str): Name of the disease being diagnosed
    """
    print("\n" + "="*50)
    print(f"DIAGNOSTIC REPORT: {disease_name.upper()}")
    print("="*50)
    
    probability = report['disease_probability']
    print(f"Probability: {probability:.2%}")
    
    risk_level = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.3 else "LOW"
    print(f"Risk Level: {risk_level}")
    
    print("\nTop Contributing Factors:")
    for factor, contribution in report['contributing_factors'].items():
        direction = "INCREASES" if contribution > 0 else "DECREASES"
        print(f"- {factor}: {direction} risk (contribution: {contribution:.4f})")
    
    print("\nRECOMMENDED ACTION:")
    if probability >= 0.7:
        print("Immediate medical attention recommended.")
    elif probability >= 0.3:
        print("Further testing recommended.")
    else:
        print("Monitor symptoms and follow up as needed.")
    
    print("="*50)

def run_disease_prediction_workflow(data_path, target_disease_column, categorical_cols=None, numeric_cols=None):
    """
    Run the complete workflow for disease prediction
    
    Parameters:
    data_path (str): Path to the dataset
    target_disease_column (str): Name of the column indicating presence of disease
    categorical_cols (list): List of categorical columns
    numeric_cols (list): List of numeric columns to standardize
    
    Returns:
    tuple: (model, trace, coefficients, X_test, feature_names)
    """
    # Load data
    print("\nSTEP 1: Loading Data")
    print("-"*50)
    df = load_data(data_path)
    if df is None:
        return None
    
    # Explore data
    print("\nSTEP 2: Exploring Data")
    print("-"*50)
    stats = explore_data(df)
    
    # If categorical and numeric columns are not provided, make an educated guess
    if categorical_cols is None:
        categorical_cols = stats['categorical_columns']
    
    if numeric_cols is None:
        numeric_cols = stats['numeric_columns']
    
    # Preprocess data
    print("\nSTEP 3: Preprocessing Data")
    print("-"*50)
    X_train, X_test, y_train, y_test, scaler, processed_df = preprocess_data(
        df, target_disease_column, categorical_cols, numeric_cols
    )
    
    # Get feature names after preprocessing
    feature_names = X_train.columns.tolist()
    
    # Build Bayesian model
    print("\nSTEP 4: Building Bayesian Model")
    print("-"*50)
    model, trace, coefficients = build_bayesian_model(
        X_train.values, y_train.values, feature_names
    )
    
    # Evaluate model
    print("\nSTEP 5: Evaluating Model")
    print("-"*50)
    metrics, y_pred_proba = evaluate_model(model, trace, X_test.values, y_test.values)
    
    # Visualize results
    print("\nSTEP 6: Visualizing Results")
    print("-"*50)
    visualize_results(trace, coefficients, feature_names, X_test.values, y_test.values, y_pred_proba)
    
    return model, trace, coefficients, X_test, feature_names, processed_df

def predict_for_new_patient(coefficients, feature_names, patient_data, disease_name):
    """
    Predict disease probability for a new patient
    
    Parameters:
    coefficients (dict): Model coefficients
    feature_names (list): Names of the features
    patient_data (dict): Dictionary of patient data (feature name -> value)
    disease_name (str): Name of the disease
    
    Returns:
    dict: Diagnostic report
    """
    # Convert patient data to the correct format
    patient_array = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        if feature in patient_data:
            patient_array[i] = patient_data[feature]
    
    # Generate report
    report = generate_diagnostic_report(coefficients, patient_array, feature_names)
    print_diagnostic_report(report, disease_name)
    
    return report

# Main execution function
def main():
    """Main function to execute the diagnostic workflow"""
    
    # File path to the dataset
    file_path = r"C:\Users\emese\Desktop\TreatmentFlow\BayesNets\symbipredict_2022.csv"
    
    # Example execution of the workflow 
    # You'll need to replace 'disease_target' with your actual target column name
    # and update categorical_cols and numeric_cols based on your data
    
    print("="*80)
    print("TREATMENT FLOW: BAYESIAN NETWORK DISEASE DIAGNOSTIC SYSTEM")
    print("="*80)
    
    # Run the workflow
    # Note: This is a placeholder - you should update these parameters based on your actual dataset
    model, trace, coefficients, X_test, feature_names, processed_df = run_disease_prediction_workflow(
        data_path=file_path,
        target_disease_column='disease_target',  # Replace with actual target column
        categorical_cols=None,  # Will be auto-detected if None
        numeric_cols=None       # Will be auto-detected if None
    )
    
    # Example: Predict for a new patient
    # Note: This is a placeholder - you should update with actual patient data
    print("\nSTEP 7: Generating Sample Diagnostic Report")
    print("-"*50)
    
    # Example patient data (replace with actual features from your model)
    sample_patient = {}
    for feature in feature_names:
        # Use random data for demonstration
        # In a real application, this would come from the patient's triage data
        if 'age' in feature.lower():
            sample_patient[feature] = 65  # Example age
        elif 'gender' in feature.lower() or 'sex' in feature.lower():
            sample_patient[feature] = 1   # Example gender encoding
        else:
            # Random value for other features
            sample_patient[feature] = np.random.randint(0, 2)
    
    # Generate diagnostic report
    predict_for_new_patient(
        coefficients, 
        feature_names, 
        sample_patient, 
        "Example Disease"  # Replace with actual disease name
    )
    
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()