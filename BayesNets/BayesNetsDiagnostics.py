import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Load and preprocess the data from the CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, all_columns, encoded_diagnosis)
    """
    print(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Check for any missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Dropping rows with missing values.")
        df = df.dropna()
    
    # Get diagnosis column name (assumes it's the last column)
    target_col = 'prognosis'
    if target_col not in df.columns:
        # Try to find a column that might represent diagnosis
        potential_targets = [col for col in df.columns if 'diagnosis' in col.lower() 
                           or 'prognosis' in col.lower() 
                           or 'disease' in col.lower()]
        if potential_targets:
            target_col = potential_targets[0]
            print(f"Using {target_col} as the target variable")
        else:
            # Assume the last column is the target
            target_col = df.columns[-1]
            print(f"No clear target column found. Using the last column ({target_col}) as target")
    
    # Create a mapping for diagnoses
    unique_diagnoses = df[target_col].unique()
    diagnosis_to_code = {diagnosis: i for i, diagnosis in enumerate(unique_diagnoses)}
    code_to_diagnosis = {i: diagnosis for i, diagnosis in enumerate(unique_diagnoses)}
    
    # Encode the target variable
    df['encoded_diagnosis'] = df[target_col].map(diagnosis_to_code)
    
    # Prepare features and target
    X = df.drop([target_col, 'encoded_diagnosis'], axis=1)
    y = df['encoded_diagnosis']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                       random_state=random_state, 
                                                       stratify=y)
    
    # Convert to pandas dataframes
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    # Combine features and target for Bayesian Network
    train_data = X_train.copy()
    train_data['diagnosis'] = y_train
    
    # Get all column names
    all_columns = list(train_data.columns)
    
    print(f"Data loaded successfully. Training set: {train_data.shape}, Test set: {X_test.shape}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Number of classes: {len(unique_diagnoses)}")
    
    return train_data, X_test, y_test, all_columns, code_to_diagnosis


def learn_structure(data, scoring_method='bic', max_indegree=3, max_iter=1000):
    """
    Learn the structure of the Bayesian Network.
    
    Args:
        data (pd.DataFrame): Training data
        scoring_method (str): Scoring method to use ('bic' or 'k2')
        max_indegree (int): Maximum number of parents for each node
        max_iter (int): Maximum number of iterations
        
    Returns:
        pgmpy.models.BayesianModel: The learned structure
    """
    print(f"Learning network structure using {scoring_method} scoring method...")
    
    # Convert categorical variables to integers if needed
    data_copy = data.copy()
    for col in data_copy.columns:
        if data_copy[col].dtype == 'object' or data_copy[col].dtype.name == 'category':
            data_copy[col] = data_copy[col].astype('category').cat.codes
    
    # Select scoring method
    if scoring_method.lower() == 'k2':
        scoring_method = K2Score(data_copy)
    else:  # Default to BIC
        scoring_method = BicScore(data_copy)
    
    # Learn structure
    hc = HillClimbSearch(data_copy)
    model_structure = hc.estimate(
        scoring_method=scoring_method,
        max_indegree=max_indegree,
        max_iter=max_iter
    )
    
    print(f"Structure learning complete. Found {len(model_structure.edges())} edges.")
    return model_structure


def build_and_train_model(structure, data, prior_type="BDeu", equivalent_sample_size=5):
    """
    Build and train the Bayesian Network model.
    
    Args:
        structure: The structure of the Bayesian Network
        data (pd.DataFrame): Training data
        prior_type (str): Type of prior to use
        equivalent_sample_size (int): Equivalent sample size for BDeu prior
        
    Returns:
        pgmpy.models.BayesianNetwork: The trained model
    """
    print("Building and training the model...")
    
    # Convert categorical variables to integers if needed
    data_copy = data.copy()
    for col in data_copy.columns:
        if data_copy[col].dtype == 'object' or data_copy[col].dtype.name == 'category':
            data_copy[col] = data_copy[col].astype('category').cat.codes
    
    # Create the model
    model = BayesianNetwork(structure.edges())
    
    # Add CPDs (Conditional Probability Distributions)
    model.fit(
        data_copy, 
        estimator=BayesianEstimator, 
        prior_type=prior_type, 
        equivalent_sample_size=equivalent_sample_size
    )
    
    print("Model built and trained successfully.")
    return model


def visualize_network(model, save_path=None, figsize=(12, 10)):
    """
    Visualize the Bayesian Network.
    
    Args:
        model: The Bayesian Network model
        save_path (str): Path to save the visualization
        figsize (tuple): Figure size
        
    Returns:
        None
    """
    print("Visualizing the Bayesian Network...")
    
    plt.figure(figsize=figsize)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges from the model
    for edge in model.edges():
        G.add_edge(edge[0], edge[1])
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    nx.draw(
        G, 
        pos, 
        with_labels=True, 
        node_color='lightblue',
        node_size=1500, 
        arrowsize=20, 
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        width=1.5
    )
    
    plt.title('Bayesian Network Structure', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test, diagnosis_map):
    """
    Evaluate the Bayesian Network model.
    
    Args:
        model: The Bayesian Network model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        diagnosis_map (dict): Mapping from code to diagnosis
        
    Returns:
        float: Accuracy score
    """
    print("Evaluating the model...")
    
    # Create an inference object
    inference = VariableElimination(model)
    
    # Make predictions
    predictions = []
    
    for i in range(len(X_test)):
        # Get evidence
        evidence = X_test.iloc[i].to_dict()
        
        # Ensure all values are integers
        evidence = {k: int(v) for k, v in evidence.items()}
        
        # Make prediction
        try:
            pred = inference.map_query(variables=['diagnosis'], evidence=evidence)
            predictions.append(pred['diagnosis'])
        except Exception as e:
            print(f"Error during prediction: {e}")
            # If prediction fails, predict the most common class
            predictions.append(y_test.mode()[0])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Convert numerical labels to original diagnoses for better interpretability
    y_test_names = [diagnosis_map[code] for code in y_test]
    pred_names = [diagnosis_map[code] for code in predictions]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_names, pred_names))
    
    # Plot confusion matrix for top classes
    plt.figure(figsize=(10, 8))
    
    # Get top 10 classes for confusion matrix (if there are more than 10)
    if len(set(y_test)) > 10:
        # Get top 10 most frequent classes
        top_classes = y_test.value_counts().nlargest(10).index
        mask_test = y_test.isin(top_classes)
        cm = confusion_matrix(
            [y_test_names[i] for i in range(len(y_test)) if mask_test.iloc[i]], 
            [pred_names[i] for i in range(len(predictions)) if mask_test.iloc[i]]
        )
        class_names = [diagnosis_map[code] for code in top_classes]
    else:
        cm = confusion_matrix(y_test_names, pred_names)
        class_names = [diagnosis_map[code] for code in sorted(set(y_test))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return accuracy


def predict_diagnosis(model, evidence, target_variable="diagnosis"):
    """
    Predict diagnosis given evidence.
    
    Args:
        model: The Bayesian Network model
        evidence (dict): Evidence for prediction
        target_variable (str): Target variable name
        
    Returns:
        dict: Prediction result
    """
    # Ensure all values are integers
    evidence = {k: int(v) for k, v in evidence.items()}
    
    # Create an inference object
    infer = VariableElimination(model)
    
    # Make prediction
    prediction = infer.map_query(variables=[target_variable], evidence=evidence)
    
    return prediction


def identify_important_features(model, data, top_n=10):
    """
    Identify important features in the Bayesian Network.
    
    Args:
        model: The Bayesian Network model
        data (pd.DataFrame): Data used to build the model
        top_n (int): Number of top features to display
        
    Returns:
        list: List of important features
    """
    print("Identifying important features...")
    
    # Get node degrees (number of connections)
    G = nx.DiGraph()
    for edge in model.edges():
        G.add_edge(edge[0], edge[1])
    
    # Get degree centrality
    centrality = nx.degree_centrality(G)
    
    # Sort features by centrality
    sorted_features = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Display top features
    print("\nTop features by network centrality:")
    for feature, score in sorted_features[:top_n]:
        if feature != 'diagnosis':  # Exclude the target variable
            print(f"{feature}: {score:.4f}")
    
    # Return important features (excluding the target variable)
    return [f for f, _ in sorted_features if f != 'diagnosis']


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define file path
    file_path = "../TreatmentFlow/BayesNets/symbipredict_2022.csv"
    
    # Load and preprocess the data
    train_data, X_test, y_test, all_columns, diagnosis_map = load_and_preprocess_data(file_path)
    
    # Learn the Bayesian Network structure
    structure = learn_structure(train_data, scoring_method='bic', max_indegree=5)
    
    # Build and train the model
    model = build_and_train_model(structure, train_data)
    
    # Visualize the network
    visualize_network(model, save_path="bayesian_network.png")
    
    # Identify important features
    important_features = identify_important_features(model, train_data)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test, diagnosis_map)
    
    # Example prediction
    print("\nExample prediction:")
    # Randomly select a test case
    test_case = X_test.sample(1).iloc[0]
    evidence = test_case.to_dict()
    
    # Make prediction
    prediction = predict_diagnosis(model, evidence)
    predicted_code = prediction['diagnosis']
    predicted_diagnosis = diagnosis_map[predicted_code]
    
    print(f"Evidence: {evidence}")
    print(f"Predicted diagnosis: {predicted_diagnosis}")
    
    # Compare with actual diagnosis
    actual_code = y_test.iloc[X_test.index.get_loc(test_case.name)]
    actual_diagnosis = diagnosis_map[actual_code]
    print(f"Actual diagnosis: {actual_diagnosis}")
    
    print("\nBayesian Network analysis complete!")


if __name__ == "__main__":
    main()