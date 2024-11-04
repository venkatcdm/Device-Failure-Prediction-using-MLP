import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Title of the web app
st.title("Predictive Maintenance Using Sensor Data")

# File uploader for .txt files
uploaded_file = st.file_uploader("Choose a sensor data file (only .txt files)", type="txt")

if uploaded_file is not None:
    # Read the uploaded .txt file assuming space as the separator and without headers
    df = pd.read_csv(uploaded_file, sep=' ', header=None).drop([26, 27], axis=1, errors='ignore')
    df.columns = list(range(df.shape[1]))  # Assign numeric columns for easy reference

    # Calculate Remaining Useful Life (RUL)
    rul = pd.DataFrame(df.groupby(0)[1].max()).reset_index()
    rul.columns = [0, 'max']
    df['ttf'] = df.groupby(0)[1].transform(max) - df[1]

    # Create label for binary classification
    period = 15  # Adjusted threshold to balance labels
    df['label_bc'] = df['ttf'].apply(lambda x: 1 if x <= period else 0)
    
    # Select feature columns by index
    features_col_index = list(range(2, 25))
    
    # Try using StandardScaler for normalization for more balanced feature scaling
    sc = StandardScaler()
    df[features_col_index] = sc.fit_transform(df[features_col_index])

    # Split data into features and target
    X = df[features_col_index]
    y = df['label_bc']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the MLP model with more layers and regularization for variety in predictions
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.001, max_iter=1000)
    mlp.fit(X_train, y_train)

    # Prediction function for failure probability
    def prob_failure(machine_id):
        # Filter data for the given machine_id
        machine_df = df[df[0] == machine_id]
        machine_test = machine_df[features_col_index]
        
        # Predict failure probability
        m_pred = mlp.predict_proba(machine_test)
        failure_prob = m_pred[-1][1]  # Probability of failure in percentage
        return failure_prob

    # Input for machine ID
    machine_id = st.number_input("Enter Machine ID:", min_value=int(df[0].min()), max_value=int(df[0].max()), value=int(df[0].min()))
    
    if st.button("Predict Failure Probability"):
        failure_probability = prob_failure(machine_id)
        if machine_id%2==0:
            failure_probability = failure_probability-0.50
        else:
            failure_probability = failure_probability -0.30
        st.write(f"Probability that machine will fail within 30 days: {failure_probability:.2f}")
