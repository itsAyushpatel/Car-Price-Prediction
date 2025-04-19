import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Print initial data overview
    print("Original data shape:", df.shape)
    
    # Drop unnecessary columns if any
    if 'PostedDate' in df.columns:
        df = df.drop(['PostedDate', 'AdditionInfo'], axis=1, errors='ignore')
    
    # Handle NA values
    df = df.dropna()
    
    # Process AskPrice column (remove ₹ symbol and commas, convert to float)
    df['AskPrice'] = df['AskPrice'].replace('[\₹,]', '', regex=True).astype(float)
    
    # Process kmDriven column (remove 'km' and commas, convert to float)
    df['kmDriven'] = df['kmDriven'].astype(str).replace('[ km,]', '', regex=True).astype(float)
    
    # Ensure Year and Age are integers
    df['Year'] = df['Year'].astype(int)
    df['Age'] = df['Age'].astype(int)
    
    # Handle additional data quality issues
    # Remove extreme outliers based on domain knowledge
    df = df[(df['AskPrice'] > 50000) & (df['AskPrice'] < 10000000)]  # reasonable car price range in INR
    df = df[(df['kmDriven'] > 1000) & (df['kmDriven'] < 500000)]  # reasonable km range
    
    # Combine Brand and model to create a more specific model feature
    df['BrandModel'] = df['Brand'] + '_' + df['model']
    
    return df

# Function to prepare data for modeling
def prepare_data_for_modeling(df):
    # Define features and target
    X = df.drop(['AskPrice', 'model'], axis=1)  # Drop model since we have BrandModel
    y = df['AskPrice']
    
    # Define categorical and numerical features
    categorical_features = ['Brand', 'Transmission', 'Owner', 'FuelType', 'BrandModel']
    numerical_features = ['Year', 'Age', 'kmDriven']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor, categorical_features, numerical_features

# Function to split data into training and testing sets
def split_data(X, y, train_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train regression models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        # Create and fit pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")
        
        results[name] = {'rmse': rmse, 'r2': r2, 'mae': mae, 'predictions': y_pred}
        fitted_models[name] = pipeline
    
    # Find best model based on RMSE
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    print(f"Best model: {best_model_name}")
    
    return best_model_name, fitted_models[best_model_name], results

# Function to visualize actual vs predicted prices
def visualize_predictions(y_test, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    
    # Save the plot temporarily
    plot_path = 'actual_vs_predicted.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

# Main function for the Streamlit app
def main():
    st.title("Used Car Price Prediction")
    st.write("This app predicts the price of a used car based on its specifications.")
    
    # Load demo dataset for initial model
    demo_data_path = "used_car_dataset.csv"  # Ensure this file exists in your working directory
    
    try:
        # Load and preprocess the data
        df = load_and_preprocess_data(demo_data_path)
        
        # Extract brand options from the data
        brands = sorted(df['Brand'].unique())
        transmissions = sorted(df['Transmission'].unique())
        owners = sorted(df['Owner'].unique())
        fuel_types = sorted(df['FuelType'].unique())
        
        year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
        km_min, km_max = int(df['kmDriven'].min()), int(df['kmDriven'].max())
        
        # Prepare data for modeling
        X, y, preprocessor, cat_features, num_features = prepare_data_for_modeling(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train models (do this once at startup)
        with st.spinner("Training models... (this might take a moment)"):
            best_model_name, best_model, results = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, preprocessor
            )
            
            # Create visualization
            plot_path = visualize_predictions(
                y_test, results[best_model_name]['predictions'], 
                f"{best_model_name} - Actual vs Predicted Prices"
            )
        
        # Model performance metrics
        st.subheader("Model Performance")
        st.write(f"Best Model: {best_model_name}")
        metrics = results[best_model_name]
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"₹{metrics['rmse']:,.2f}")
        col2.metric("R²", f"{metrics['r2']:.2f}")
        col3.metric("MAE", f"₹{metrics['mae']:,.2f}")
        
        # Display the actual vs predicted plot
        st.image(plot_path)
        
        # Input form for prediction
        st.subheader("Predict Car Price")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", brands)
            transmission = st.selectbox("Transmission", transmissions)
            owner = st.selectbox("Owner", owners)
            
        with col2:
            fuel_type = st.selectbox("Fuel Type", fuel_types)
            year = st.slider("Year", year_min, year_max, int((year_min + year_max) / 2))
            kmDriven = st.number_input("Kilometers Driven", min_value=float(km_min), 
                                      max_value=float(km_max), 
                                      value=float((km_min + km_max) / 2),
                                      step=1000.0)
        
        # Get available models for the selected brand
        available_models = df[df['Brand'] == brand]['model'].unique()
        
        # Add model selection if models are available
        if len(available_models) > 0:
            model = st.selectbox("Model", sorted(available_models))
        else:
            model = "Unknown"
        
        # Make prediction when button is clicked
        if st.button("Predict Price"):
            # Calculate age based on year
            age = 2025 - year  # Current year from the app
            
            # Create input dataframe for prediction
            input_data = pd.DataFrame({
                'Brand': [brand],
                'model': [model],
                'Year': [year],
                'Age': [age],
                'kmDriven': [kmDriven],
                'Transmission': [transmission],
                'Owner': [owner],
                'FuelType': [fuel_type],
                'BrandModel': [f"{brand}_{model}"]
            })
            
            # Make prediction using the best model
            prediction = best_model.predict(input_data)[0]
            
            # Format predicted price with currency symbol
            formatted_price = f"₹ {prediction:,.2f}"
            
            # Display prediction with larger font
            st.markdown(f"<h3 style='text-align:center;'>Predicted Price: {formatted_price}</h3>", 
                        unsafe_allow_html=True)
            
            # Display age for reference
            st.info(f"Car Age: {age} years")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please ensure you have the CSV file with car data in the correct format.")

if __name__ == "__main__":
    main()
