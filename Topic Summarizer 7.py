import pandas as pd
import torch
import joblib
import os
import warnings
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL_SAVE_DIR = "summarization_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """Initialize summarization pipeline with error handling"""
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
    def initialize(self):
        """Load model and tokenizer"""
        try:
            st.info(f"🚀 Initializing {self.model_name} summarizer...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt"
            )
            st.success("✅ Summarizer initialized successfully")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing model: {str(e)}")
            return False
    
    def summarize_text(self, text, max_length=130, min_length=30):
        """Safe text summarization with error handling"""
        try:
            if not text or pd.isna(text) or len(str(text).strip()) < 10:
                return ""
            
            # Clean text and summarize
            text = ' '.join(str(text).split())
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            st.warning(f"⚠️ Error summarizing text: {str(e)}")
            return ""
    
    def batch_summarize(self, texts, max_workers=4):
        """Process multiple texts in parallel"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.summarize_text, text) for text in texts]
            summaries = []
            for future in tqdm(as_completed(futures), total=len(texts)):
                summaries.append(future.result())
            return summaries

def process_dataset(df, summarizer, text_column, sample_size=None, batch_size=4):
    """Process dataframe with progress tracking"""
    if sample_size and sample_size > 0:
        df = df.sample(min(sample_size, len(df)))
    
    st.info(f"📊 Processing {len(df)} texts...")
    
    # Process in batches if batch_size > 1
    if batch_size > 1:
        texts = df[text_column].tolist()
        summaries = summarizer.batch_summarize(texts, max_workers=batch_size)
        df['summary'] = summaries
    else:
        progress_bar = st.progress(0)
        summaries = []
        for i, text in enumerate(df[text_column]):
            summaries.append(summarizer.summarize_text(text))
            progress_bar.progress((i + 1) / len(df))
        df['summary'] = summaries
    
    return df

def main():
    st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
    st.title("📝 AI Text Summarizer")
    st.markdown("Summarize text using advanced NLP models like BART")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select Model",
            ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"],
            index=0
        )
        max_length = st.slider("Max Summary Length", 50, 200, 130)
        min_length = st.slider("Min Summary Length", 10, 100, 30)
        batch_size = st.slider("Batch Size (for files)", 1, 8, 2)
        sample_size = st.number_input("Sample Size (0 for all)", min_value=0, value=0)
    
    # File upload or text input
    input_option = st.radio("Input Type", ["Text Input", "File Upload"])
    
    if input_option == "Text Input":
        text = st.text_area("Enter text to summarize", height=200)
        if st.button("Summarize"):
            if text.strip():
                summarizer = TextSummarizer(model_name)
                if summarizer.initialize():
                    summary = summarizer.summarize_text(text, max_length, min_length)
                    st.subheader("Summary")
                    st.write(summary)
            else:
                st.warning("Please enter some text to summarize")
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded {len(df)} records")
                
                # Let user select which column contains text to summarize
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                if not text_columns:
                    st.error("No text columns found in the uploaded file.")
                    return
                
                text_column = st.selectbox("Select column containing text to summarize", text_columns)
                
                st.dataframe(df.head())
                
                if st.button("Process File"):
                    summarizer = TextSummarizer(model_name)
                    if summarizer.initialize():
                        with st.spinner("Processing..."):
                            result_df = process_dataset(
                                df, 
                                summarizer, 
                                text_column=text_column,
                                sample_size=sample_size if sample_size > 0 else None, 
                                batch_size=batch_size
                            )
                        
                        st.subheader("Results")
                        st.dataframe(result_df)
                        
                        # Download button
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="summarized_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
