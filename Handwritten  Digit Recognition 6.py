import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import streamlit as st
import warnings
import os
import ast

warnings.filterwarnings("ignore")

# Set matplotlib to use Agg backend to prevent GUI issues
plt.switch_backend('Agg')

# 1. Function to load and preprocess the custom MNIST dataset from CSV
def load_custom_mnist(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Convert string representation of image arrays to actual numpy arrays
        df['image'] = df['image'].apply(lambda x: np.array(ast.literal_eval(x)))
        
        # Convert to numpy arrays
        images = np.stack(df['image'].values)
        labels = df['label'].values
        
        # Normalize pixel values to 0-1 range
        images = images / 255.0
        
        # Add channel dimension (for CNN)
        images = np.expand_dims(images, axis=-1)
        
        return images, labels
    except Exception as e:
        st.error(f"Error loading custom MNIST data: {e}")
        return None, None

# 2. Function to create the CNN model
def create_cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

# 3. Function to load the trained model
def load_cnn_model(model_path='mnist_cnn_model.keras'):
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# 4. Streamlit app
def main():
    st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")
    st.title("Handwritten Digit Recognition")
    st.write("This model recognizes handwritten digits from images using a CNN.")

    # Initialize session state for model and dataset
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False

    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        dataset_option = st.radio("Choose dataset:", ("Built-in MNIST", "Custom CSV"))
        
        if dataset_option == "Custom CSV":
            csv_path = st.text_input("Enter path to CSV dataset", 
                                   r'C:\Users\Mogeeth.M\Downloads\chill\chill\Digital_recognition6.csv')

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Dataset loading section
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                if dataset_option == "Built-in MNIST":
                    mnist = keras.datasets.mnist
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
                    x_train, x_test = x_train / 255.0, x_test / 255.0
                    x_train = np.expand_dims(x_train, axis=-1)
                    x_test = np.expand_dims(x_test, axis=-1)
                    
                    st.session_state.x_train = x_train
                    st.session_state.y_train = y_train
                    st.session_state.x_test = x_test
                    st.session_state.y_test = y_test
                    
                    st.success("MNIST dataset loaded successfully!")
                    
                    # Show dataset info
                    st.write(f"Training set shape: {x_train.shape}")
                    st.write(f"Test set shape: {x_test.shape}")
                    st.write(f"Number of classes: {len(np.unique(y_train))}")
                    
                    # Show sample images
                    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(x_train[i].squeeze(), cmap='gray')
                        ax.set_title(f"Label: {y_train[i]}")
                        ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show class distribution
                    fig = plt.figure(figsize=(8, 4))
                    sns.countplot(x=y_train)
                    plt.title("Class Distribution in Training Set")
                    plt.xlabel("Digit Label")
                    plt.ylabel("Count")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                else:
                    if not csv_path:
                        st.warning("Please enter a valid CSV file path")
                        return
                        
                    x_train, y_train = load_custom_mnist(csv_path)
                    if x_train is None or y_train is None:
                        return
                        
                    # Split into train and test sets
                    split_ratio = 0.8
                    split_idx = int(len(x_train) * split_ratio)
                    st.session_state.x_train = x_train[:split_idx]
                    st.session_state.x_test = x_train[split_idx:]
                    st.session_state.y_train = y_train[:split_idx]
                    st.session_state.y_test = y_train[split_idx:]
                    
                    st.success("Custom dataset loaded successfully!")
                
                st.session_state.dataset_loaded = True

        # Model training section
        if st.session_state.dataset_loaded and st.button("Train Model"):
            with st.spinner("Creating and training model..."):
                model = create_cnn_model()
                history = model.fit(
                    st.session_state.x_train, 
                    st.session_state.y_train, 
                    epochs=5, 
                    validation_data=(st.session_state.x_test, st.session_state.y_test), 
                    verbose=0
                )
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history.history['accuracy'], label='train accuracy')
                ax1.plot(history.history['val_accuracy'], label='validation accuracy')
                ax1.set_title('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history['loss'], label='train loss')
                ax2.plot(history.history['val_loss'], label='validation loss')
                ax2.set_title('Loss')
                ax2.legend()
                st.pyplot(fig)
                plt.close(fig)
                
                # Evaluate model
                test_loss, test_acc = model.evaluate(
                    st.session_state.x_test, 
                    st.session_state.y_test, 
                    verbose=0
                )
                st.write(f"Test Accuracy: {test_acc:.4f}")
                st.write(f"Test Loss: {test_loss:.4f}")
                
                # Save model and update session state
                model.save('mnist_cnn_model.keras')
                st.session_state.model = model
                st.success("Model trained and saved successfully!")
                
                # Show classification report and confusion matrix
                y_pred = model.predict(st.session_state.x_test, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                st.subheader("Classification Report")
                st.text(classification_report(st.session_state.y_test, y_pred_classes))
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, y_pred_classes)
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                            xticklabels=range(10), yticklabels=range(10))
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)
                plt.close(fig)

    with col2:
        # Prediction section
        st.header("Try It Out")
        uploaded_file = st.file_uploader(
            "Upload an image of a handwritten digit", 
            type=["png", "jpg", "jpeg"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            try:
                # Load and preprocess the image
                with st.spinner("Processing image..."):
                    image = Image.open(uploaded_file).convert("L")
                    st.image(image, caption='Uploaded Image', width=200)
                    
                    # Resize and normalize
                    image = image.resize((28, 28))
                    image_array = np.array(image) / 255.0
                    image_array = image_array.reshape(1, 28, 28, 1)

                # Load or get model from session state
                if st.session_state.model is None:
                    st.session_state.model = load_cnn_model()
                    
                if st.session_state.model is None:
                    st.error("No model available. Please train a model first.")
                    return

                # Make prediction
                with st.spinner("Predicting digit..."):
                    prediction = st.session_state.model.predict(image_array, verbose=0)
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    st.success(f"🔢 **Predicted Digit:** `{digit}`")
                    st.success(f"🎯 **Confidence:** {confidence:.2%}")
                    
                    # Show prediction probabilities
                    st.subheader("Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': prediction[0]
                    })
                    st.bar_chart(prob_df.set_index('Digit'))
                    
                    # Visualize the processed image
                    st.subheader("Processed Image")
                    fig = plt.figure(figsize=(3, 3))
                    plt.imshow(image_array.squeeze(), cmap='gray')
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Please upload an image to begin digit recognition.")

if __name__ == "__main__":
    main()
