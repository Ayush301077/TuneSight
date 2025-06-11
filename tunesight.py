import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="ML Hyperparameter Tuning", layout="wide")

def load_data():
    """Load and preprocess data"""
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def explore_data(df):
    """Data exploration and visualization"""
    st.subheader("📊 Data Exploration")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numerical Cols", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

def main():
    st.title("🤖 Machine Learning Hyperparameter Tuning App")
    st.markdown("Upload your dataset and let's find the best model with optimal parameters!")

    df = load_data()

    if df is not None:
        explore_data(df)

if __name__ == "__main__":
    main()
