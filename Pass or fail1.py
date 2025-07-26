import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Mogeeth.M\Downloads\chill\chill\passor_fail1.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    # Convert correct to binary (1 for correct, 0 for incorrect)
    df['is_correct'] = df['correct'].astype(int)
    
    # Calculate pass/fail at student level (threshold can be adjusted)
    student_stats = df.groupby('student_id').agg({
        'is_correct': 'mean',
        'time_sec': 'mean',
        'attempt_count': 'mean',
        'hint_count': 'mean'
    }).reset_index()
    
    # Define pass/fail (threshold = 0.7 correct answers)
    student_stats['pass'] = (student_stats['is_correct'] >= 0.7).astype(int)
    
    # Get skill features for each student
    student_skills = df.groupby(['student_id', 'skill_id']).size().unstack(fill_value=0)
    
    # Merge all features
    final_df = student_stats.merge(student_skills, on='student_id')
    
    return final_df

# Train model
def train_model(df):
    # Features and target
    X = df.drop(['student_id', 'pass', 'is_correct'], axis=1)
    y = df['pass']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessing for different feature types
    numeric_features = ['time_sec', 'attempt_count', 'hint_count']
    skill_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('skills', 'passthrough', skill_features)
        ])
    
    # Gradient Boosting Classifier
    model = make_pipeline(
        preprocessor,
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report, X.columns

# Feature importance visualization
def plot_feature_importance(model, feature_names):
    gb = model.named_steps['gradientboostingclassifier']
    importances = gb.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    st.pyplot(plt)

# Main app
def main():
    st.title("🎓 Student Pass/Fail Prediction")
    st.write("""
    This app predicts whether a student will pass or fail based on their:
    - Historical performance
    - Time spent on problems
    - Attempt counts
    - Hint usage
    - Skill mastery
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.write(df.head())
    
    # Preprocess data
    with st.spinner("Processing data..."):
        processed_df = preprocess_data(df)
    
    # Train model
    with st.spinner("Training model..."):
        model, accuracy, report, feature_names = train_model(processed_df)
    
    st.success("Model trained successfully!")
    st.write(f"Model Accuracy: {accuracy:.2%}")
    
    # Show classification report
    st.subheader("Model Performance")
    st.text(report)
    
    # Feature importance
    st.subheader("Key Influencing Features")
    plot_feature_importance(model, feature_names)
    
    # Prediction interface
    st.subheader("Make a Prediction")
    
    # Get median values for reference
    time_median = processed_df['time_sec'].median()
    attempt_median = processed_df['attempt_count'].median()
    hint_median = processed_df['hint_count'].median()
    
    col1, col2 = st.columns(2)
    with col1:
        time_spent = st.number_input("Average Time Spent (seconds)", 
                                   min_value=0, 
                                   max_value=int(processed_df['time_sec'].max()), 
                                   value=int(time_median))
        attempts = st.number_input("Average Attempt Count", 
                                 min_value=1, 
                                 max_value=int(processed_df['attempt_count'].max()), 
                                 value=int(attempt_median))
    with col2:
        hints = st.number_input("Average Hint Count", 
                              min_value=0, 
                              max_value=int(processed_df['hint_count'].max()), 
                              value=int(hint_median))
        skill = st.selectbox("Most Practiced Skill", df['skill_id'].unique())
    
    if st.button("Predict Pass/Fail"):
        # Create input features with ALL features the model expects
        input_data = {
            'time_sec': [time_spent],
            'attempt_count': [attempts],
            'hint_count': [hints]
        }
        
        # Initialize all skills to 0
        for sk in df['skill_id'].unique():
            input_data[sk] = [0]
        
        # Set the selected skill to 1
        input_data[skill] = [1]
        
        # Create DataFrame with columns in same order as training data
        input_df = pd.DataFrame(input_data)[feature_names]
        
        # Make prediction
        try:
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)
            
            if prediction[0] == 1:
                st.success(f"Prediction: PASS (Probability: {proba[0][1]:.2%})")
            else:
                st.error(f"Prediction: FAIL (Probability: {proba[0][0]:.2%})")
            
            # Show feature contributions
            st.write("### Key Factors in This Prediction:")
            st.write(f"- Time spent: {'High' if time_spent > time_median else 'Low'} (Median: {time_median:.1f} sec)")
            st.write(f"- Attempts: {'High' if attempts > attempt_median else 'Low'} (Median: {attempt_median:.1f})")
            st.write(f"- Hints used: {'High' if hints > hint_median else 'Low'} (Median: {hint_median:.1f})")
            st.write(f"- Skill practiced: {skill}")
            
            # Show passing tips if prediction is FAIL
            if prediction[0] == 0:
                st.warning("""
                **Tips to Improve:**
                - Reduce time spent per problem (aim for 30-45 seconds)
                - Try solving without hints
                - Focus on high-impact skills (see Feature Importance above)
                - Review incorrect answers to identify weak areas
                """)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
