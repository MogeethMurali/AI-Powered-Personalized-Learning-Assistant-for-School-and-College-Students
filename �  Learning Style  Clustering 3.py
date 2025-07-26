import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Student Learning Style Clustering", layout="wide")

# Cache expensive computations
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def preprocess_data(df, features):
    X = df[features]
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

@st.cache_data
def perform_clustering(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    return cluster_labels, kmeans, silhouette_avg

@st.cache_data
def perform_pca(X_scaled):
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)

def process_single_student(input_text, scaler, kmeans):
    components = [x.strip() for x in input_text.split(',')]
    
    if len(components) < 12:
        raise ValueError("Not enough data points provided. Expected format: ID,Name,Age,Gender,reading_speed,comprehension_score,reading_efficiency,video_time,text_time,visual_engagement,response_time,quiz_score,preferred_style,preferred_medium")
    
    features = [
        float(components[4]),  # reading_speed
        float(components[5]),  # comprehension_score
        float(components[6]),  # reading_efficiency
        float(components[7]),  # video_time
        float(components[8]),  # text_time
        float(components[9]),  # visual_engagement
        float(components[10]), # response_time
        float(components[11])  # quiz_score
    ]
    
    # Scale the features using the pre-trained scaler
    features_scaled = scaler.transform([features])
    
    # Predict the cluster
    cluster = kmeans.predict(features_scaled)[0]
    
    return cluster, features_scaled

def main():
    st.title("📚 Student Learning Style Clustering")
    st.markdown("""
    Cluster students into groups like 'visual learners', 'slow learners', and 'fast responders' 
    based on their interaction data, reading speed, and topic-wise performance.
    """)
    
    tab1, tab2 = st.tabs(["📁 Upload CSV File", "📝 Analyze Single Student"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            
            st.subheader("📄 Raw Data Preview")
            st.dataframe(df.head())
            
            # Define features for clustering
            features = [
                'reading_speed', 'comprehension_score', 'reading_efficiency',
                'video_time', 'text_time', 'visual_engagement',
                'response_time', 'quiz_score'
            ]
            
            # Check for missing features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return
            
            # Preprocess data
            X_scaled, scaler = preprocess_data(df, features)
            
            # Determine optimal clusters using silhouette score
            st.subheader("🔍 Determine Optimal Number of Clusters")
            max_clusters = min(10, len(df)-1)
            n_clusters = st.slider("Select number of clusters", 2, max_clusters, 3)
            
            # Perform clustering
            cluster_labels, kmeans, silhouette_avg = perform_clustering(X_scaled, n_clusters)
            df['cluster'] = cluster_labels
            
            # Show silhouette score
            st.success(f"Silhouette Score: {silhouette_avg:.3f} (Higher is better)")
            
            # Define cluster labels based on characteristics
            cluster_name_mapping = {
                0: 'Visual Learners (high video_time, visual_engagement)',
                1: 'Fast Responders (low response_time, high quiz_score)',
                2: 'Slow Learners (low reading_speed, low comprehension_score)'
            }
            
            # Assign labels based on cluster numbers
            df['learning_style'] = df['cluster'].map(cluster_name_mapping)
            
            st.subheader("📌 Clustered Students")
            st.dataframe(df[['student_name', 'cluster', 'learning_style']])
            
            # Show cluster characteristics
            st.subheader("📊 Cluster Characteristics")
            cluster_means = df.groupby('cluster')[features].mean()
            st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
            
            # PCA visualization
            X_pca = perform_pca(X_scaled)
            df['pca1'] = X_pca[:, 0]
            df['pca2'] = X_pca[:, 1]
            
            st.subheader("📉 Cluster Visualization (PCA)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df, x='pca1', y='pca2', hue='learning_style',
                palette='viridis', s=100, alpha=0.8
            )
            plt.title(f"Student Learning Styles (Silhouette Score: {silhouette_avg:.2f})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            # Download results
            st.subheader("⬇️ Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Data",
                data=csv,
                file_name='student_learning_styles.csv',
                mime='text/csv'
            )
            
            # Store the trained models in session state for single student analysis
            st.session_state['scaler'] = scaler
            st.session_state['kmeans'] = kmeans
            st.session_state['cluster_names'] = cluster_name_mapping
    
    with tab2:
        st.subheader("Analyze Single Student")
        
        if 'kmeans' not in st.session_state:
            st.warning("Please first upload and analyze a dataset in the 'Upload CSV File' tab")
        else:
            input_text = st.text_area(
                "Enter student data (comma-separated values):\n"
                "Format: ID,Name,Age,Gender,reading_speed,comprehension_score,reading_efficiency,"
                "video_time,text_time,visual_engagement,response_time,quiz_score,preferred_style,preferred_medium",
                height=100
            )
            
            if st.button("Classify Learning Style"):
                if not input_text:
                    st.warning("Please enter student data")
                else:
                    try:
                        cluster, features_scaled = process_single_student(
                            input_text, 
                            st.session_state['scaler'], 
                            st.session_state['kmeans']
                        )
                        
                        # Get cluster name from session state
                        cluster_name = st.session_state['cluster_names'].get(cluster, f"Cluster {cluster}")
                        
                        # Display results
                        st.success(f"Learning Style Identified: {cluster_name}")
                        
                        # Show feature comparison
                        st.subheader("Feature Comparison to Cluster Centers")
                        cluster_centers = st.session_state['kmeans'].cluster_centers_
                        comparison_df = pd.DataFrame({
                            'Feature': features,
                            'Student Value': features_scaled[0],
                            'Cluster Center': cluster_centers[cluster]
                        })
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            data=comparison_df.melt(id_vars='Feature'), 
                            x='value', 
                            y='Feature', 
                            hue='variable',
                            orient='h'
                        )
                        plt.title("Feature Comparison with Cluster Center")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Please check the input format and try again.")

if __name__ == "__main__":
    main()
