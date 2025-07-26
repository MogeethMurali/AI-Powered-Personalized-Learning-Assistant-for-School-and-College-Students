import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import time
import os

# Set page config
st.set_page_config(
    page_title="Dropout Risk Detection",
    page_icon="⚠️",
    layout="wide"
)

# Title and description
st.title("⚠️ Dropout Risk Detection")
st.markdown("""
This application predicts which students are likely to drop out based on their activity patterns and performance metrics.
""")

# Load data function
@st.cache_data
def load_data():
    try:
        # Try to load from local path first
        data_path = r"C:\Users\Mogeeth.M\Downloads\chill\chill\Dropout Risk Detection4.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            # Fallback to relative path if absolute path fails
            data = pd.read_csv("Dropout Risk Detection4.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main function
def main():
    # Load data
    data = load_data()
    
    if data is None:
        st.stop()
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(data.head())

    # Data preprocessing
    st.subheader("Data Overview")
    
    # Check if target exists
    if 'dropout' not in data.columns:
        st.error("Target column 'dropout' not found in data")
        st.stop()
    
    # Display target distribution
    st.write("### Dropout Distribution")
    fig, ax = plt.subplots()
    data['dropout'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Distribution of Dropout Status")
    ax.set_xlabel("Dropout")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

    # Feature selection
    features = ['last_login_days', 'avg_score', 'quiz_completion_rate', 
               'forum_posts', 'video_views_per_week', 'live_session_attendance']
    
    # Check if all features exist
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        st.error(f"Missing features in data: {missing_features}")
        st.stop()
    
    X = data[features]
    y = data['dropout']

    # Model training
    st.subheader("Model Training")
    
    # Split data
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Model parameters
    st.sidebar.header("Model Parameters")
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
    n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 50)
    max_depth = st.sidebar.slider("Max depth", 3, 10, 6, 1)
    subsample = st.sidebar.slider("Subsample ratio", 0.5, 1.0, 1.0, 0.1)
    colsample_bytree = st.sidebar.slider("Column subsample ratio", 0.5, 1.0, 1.0, 0.1)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            eval_metric='auc'
        ))
    ])
    
    # Train model with progress
    with st.spinner("Training model..."):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Model evaluation
    st.subheader("Model Evaluation")
    
    # Classification report
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close()
    
    # ROC curve
    st.write("### ROC Curve")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    plt.close()
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pipeline.named_steps['xgb'].get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    plt.close()
    
    # Prediction interface
    st.subheader("Predict Dropout Risk for New Students")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            last_login = st.number_input("Days since last login", min_value=0, max_value=365, value=7)
            avg_score = st.number_input("Average score", min_value=0.0, max_value=100.0, value=75.0)
            quiz_rate = st.number_input("Quiz completion rate", min_value=0.0, max_value=1.0, value=0.8)
        
        with col2:
            forum_posts = st.number_input("Forum posts count", min_value=0, value=2)
            video_views = st.number_input("Video views per week", min_value=0, value=3)
            live_attendance = st.number_input("Live session attendance", min_value=0, value=2)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = pd.DataFrame([[last_login, avg_score, quiz_rate, forum_posts, video_views, live_attendance]],
                                    columns=features)
            
            # Scale and predict
            prediction = pipeline.predict(input_data)
            probability = pipeline.predict_proba(input_data)[0][1]
            
            if prediction[0] == 1:
                st.error(f"🚨 High dropout risk: {probability*100:.1f}% probability")
            else:
                st.success(f"✅ Low dropout risk: {(1-probability)*100:.1f}% probability")
            
            # Show probability gauge
            st.write("### Risk Probability")
            st.progress(int(probability * 100))
            st.write(f"{probability*100:.1f}% probability of dropout")

    # Download model
    if st.button("Download Model"):
        with st.spinner("Saving model..."):
            joblib.dump(pipeline, 'dropout_predictor.pkl')
            with open('dropout_predictor.pkl', 'rb') as f:
                st.download_button(
                    label="Download Trained Model",
                    data=f,
                    file_name="dropout_predictor.pkl",
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()
