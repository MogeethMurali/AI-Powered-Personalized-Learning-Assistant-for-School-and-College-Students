import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="🧾 Topic Detection from Student Answers",
    page_icon="🧾",
    layout="wide"
)

# Load model components with caching
@st.cache_resource
def load_model_components():
    try:
        model = load_model(r'C:\Users\Mogeeth.M\Downloads\chill\chill\Topic Detection5.csv\topic_classifier5.keras')
        tokenizer = load(r'C:\Users\Mogeeth.M\Downloads\chill\chill\Topic Detection5.csv\text_tokenizer5.joblib')
        le = load(r'C:\Users\Mogeeth.M\Downloads\chill\chill\Topic Detection5.csv\label_encoder5.joblib')
        return model, tokenizer, le
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# Main function
def main():
    st.title("🧾 Topic Detection from Student Answers")
    st.markdown("""
    This application uses a pre-trained LSTM model to detect the topic of student answers.
    """)

    # Load model components
    model, tokenizer, le = load_model_components()
    
    if model is None:
        st.warning("Please ensure you have the following files in your directory:")
        st.markdown("""
        - `topic_classifier5.keras` (trained LSTM model)
        - `text_tokenizer5.joblib` (text tokenizer)
        - `label_encoder5.joblib` (label encoder)
        """)
        return

    # Create tabs
    tab1, tab2 = st.tabs(["Topic Detection", "Model Information"])

    with tab1:
        st.subheader("Detect Topic from Student Answer")
        
        # Input text area
        answer = st.text_area(
            "Enter the student's answer:", 
            height=200,
            placeholder="Type or paste the student's answer here..."
        )

        if st.button("Detect Topic", type="primary"):
            if not answer.strip():
                st.warning("Please enter an answer to detect the topic.")
            else:
                with st.spinner("Analyzing the answer..."):
                    try:
                        # Preprocess the input
                        sample_seq = tokenizer.texts_to_sequences([answer])
                        sample_pad = pad_sequences(sample_seq, maxlen=100)
                        
                        # Make prediction
                        pred = model.predict(sample_pad, verbose=0)
                        pred_class = np.argmax(pred)
                        confidence = np.max(pred)
                        predicted_label = le.inverse_transform([pred_class])[0]

                        # Display results
                        st.success(f"📚 **Detected Topic:** `{predicted_label}` (Confidence: {confidence*100:.1f}%)")

                        # Show probabilities
                        st.subheader("Topic Probabilities")
                        
                        # Create dataframe with probabilities
                        prob_df = pd.DataFrame({
                            'Topic': le.classes_,
                            'Probability': pred[0]
                        }).sort_values('Probability', ascending=False)
                        
                        # Display top 5 topics
                        st.dataframe(
                            prob_df.head(5).style.format({'Probability': '{:.2%}'}),
                            height=210
                        )

                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(
                            x='Probability', 
                            y='Topic', 
                            data=prob_df.head(5), 
                            palette='viridis',
                            ax=ax
                        )
                        ax.set_title("Top 5 Predicted Topics")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

    with tab2:
        st.subheader("Model Information")
        
        # Model architecture
        st.markdown("### Model Architecture")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.text("\n".join(model_summary))
        
        # Class distribution
        st.markdown("### Available Topics")
        st.write(le.classes_)
        
        # Sample predictions
        st.markdown("### Example Predictions")
        sample_texts = [
            "The stock market reached record highs today",
            "The football team won the championship",
            "New scientific discovery in quantum physics"
        ]
        
        for text in sample_texts:
            with st.expander(f"Sample: {text[:30]}..."):
                sample_seq = tokenizer.texts_to_sequences([text])
                sample_pad = pad_sequences(sample_seq, maxlen=100)
                pred = model.predict(sample_pad, verbose=0)
                pred_class = np.argmax(pred)
                predicted_label = le.inverse_transform([pred_class])[0]
                confidence = np.max(pred)
                
                st.write(f"**Predicted Topic:** {predicted_label}")
                st.write(f"**Confidence:** {confidence*100:.1f}%")
                
                # Show probabilities
                prob_df = pd.DataFrame({
                    'Topic': le.classes_,
                    'Probability': pred[0]
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(
                    prob_df.head(5).style.format({'Probability': '{:.2%}'}),
                    height=210
                )

if __name__ == "__main__":
    main()
