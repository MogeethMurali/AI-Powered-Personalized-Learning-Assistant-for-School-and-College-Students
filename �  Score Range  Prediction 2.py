import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Mogeeth.M\Downloads\chill\chill\predict_score_range2.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    # Convert timestamp to datetime and extract features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Drop original timestamp
    df = df.drop('timestamp', axis=1)
    
    return df

# Train model
def train_model(df):
    # Features and target
    X = df.drop(['final_score', 'score_range', 'user_id', 'problem_id'], axis=1)
    y = df['final_score']
    
    # Define numeric and categorical features
    numeric_features = ['time_spent', 'difficulty', 'prev_score', 'hour', 'day_of_week']
    categorical_features = ['tag', 'bundle_id']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Ridge Regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X.columns

# Main app
def main():
    st.title("📊 Student Score Prediction")
    st.write("""
    This app predicts a student's final score (0-100) based on:
    - Time spent per question
    - Question difficulty
    - Previous scores
    - Problem topic (tag)
    - Problem bundle
    - Time of day/week
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    with st.spinner("Processing data..."):
        processed_df = preprocess_data(df)
    
    # Show EDA
    if st.checkbox("Show Exploratory Data Analysis"):
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Final Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(processed_df['final_score'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("### Score Range Distribution")
            fig, ax = plt.subplots()
            processed_df['score_range'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        
        st.write("### Numeric Features Summary")
        st.write(processed_df.describe())
        
        st.write("### Categorical Features Summary")
        st.write("Tags:", processed_df['tag'].value_counts())
        st.write("Bundles:", processed_df['bundle_id'].value_counts().head(10))
    
    # Train model
    with st.spinner("Training model..."):
        model, rmse, r2, feature_names = train_model(processed_df)
    
    st.success("Model trained successfully!")
    st.write(f"Model Performance: RMSE = {rmse:.2f}, R² = {r2:.2f}")
    
    # Prediction interface
    st.subheader("Make a Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        time_spent = st.slider("Time Spent (seconds)", 
                              min_value=5, 
                              max_value=60, 
                              value=30)
        difficulty = st.slider("Question Difficulty (1-5)", 
                              min_value=1, 
                              max_value=5, 
                              value=3)
        prev_score = st.slider("Previous Score (0-100)", 
                              min_value=0, 
                              max_value=100, 
                              value=70)
    with col2:
        hour = st.slider("Hour of Day (0-23)", 
                        min_value=0, 
                        max_value=23, 
                        value=12)
        day_of_week = st.selectbox("Day of Week", 
                                  ["Monday", "Tuesday", "Wednesday", 
                                   "Thursday", "Friday", "Saturday", "Sunday"])
        tag = st.selectbox("Problem Topic", 
                          processed_df['tag'].unique())
        bundle_id = st.selectbox("Problem Bundle", 
                                processed_df['bundle_id'].unique())
    
    # Convert day of week to numeric
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, 
               "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_num = day_map[day_of_week]
    
    if st.button("Predict Score"):
        # Create input features
        input_data = {
            'time_spent': [time_spent],
            'difficulty': [difficulty],
            'prev_score': [prev_score],
            'hour': [hour],
            'day_of_week': [day_num],
            'tag': [tag],
            'bundle_id': [bundle_id]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        try:
            prediction = model.predict(input_df)
            st.success(f"Predicted Final Score: {prediction[0]:.1f}/100")
            
            # Interpret score range
            if prediction[0] >= 80:
                st.info("Score Range: Excellent")
            elif prediction[0] >= 50:
                st.info("Score Range: Pass")
            else:
                st.warning("Score Range: Fail")
            
            # Show feature impact
            st.write("### Factors Affecting This Prediction:")
            st.write(f"- Previous Score: {prev_score} (baseline)")
            st.write(f"- Time Spent: {'Optimal' if 25 <= time_spent <= 35 else 'Suboptimal'}")
            st.write(f"- Difficulty Level: {difficulty}/5")
            st.write(f"- Problem Topic: {tag}")
            st.write(f"- Time of Day: {'Peak hours' if 9 <= hour <= 17 else 'Off-peak'}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
