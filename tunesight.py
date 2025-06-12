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
    uploaded_file = st.file_uploader("📁 Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
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

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())


def create_visualizations(df, target_col, problem_type):
    """Create data visualizations"""
    st.subheader("📉 Data Visualizations")

    # Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        st.subheader("🔗 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        plt.close()

    # Distribution plots
    st.subheader("📦 Data Distribution")
    for col in numerical_cols[:6]:
        if col != target_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].hist(bins=30, ax=ax)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            plt.close()


def prepare_data(df, target_col, test_size):
    """Prepare data for machine learning"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.mean())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if len(np.unique(y)) > 1 and len(np.unique(y)) < 20 else None
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def main():
    st.title("🤖 Machine Learning Hyperparameter Tuning App")
    st.markdown("Upload your dataset and let's explore, visualize, and prepare it for modeling!")

    df = load_data()

    if df is not None:
        explore_data(df)

        # Target variable selection
        st.subheader("🎯 Target Variable Selection")
        target_col = st.selectbox("Select the target variable:", df.columns.tolist())

        # Problem type detection
        unique_values = df[target_col].nunique()
        if unique_values <= 10 and df[target_col].dtype in ['object', 'int64']:
            problem_type = st.selectbox("📌 Choose Problem Type:", ['classification', 'regression'])
        else:
            problem_type = 'regression'
            st.info("Automatically detected problem type: **Regression**")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)

        # Visualizations
        create_visualizations(df, target_col, problem_type)

        # Prepare data
        if st.button("📦 Prepare Data"):
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(
                df, target_col, test_size
            )
            st.success("✅ Data prepared successfully!")
            st.info(f"🔹 Training set size: {X_train.shape[0]} samples")
            st.info(f"🔹 Test set size: {X_test.shape[0]} samples")


if __name__ == "__main__":
    main()
