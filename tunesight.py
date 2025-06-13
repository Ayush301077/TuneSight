import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

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

def get_param_grids():
    """Define parameter grids for hyperparameter tuning"""
    param_grids = {
        'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'lasso': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'logistic': {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        'knn_clf': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'knn_reg': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'svm_clf': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
        'svm_reg': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
        'dt_clf': {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['gini', 'entropy']},
        'dt_reg': {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'criterion': ['mse', 'mae']},
        'rf_clf': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5], 'max_features': ['auto', 'sqrt']},
        'rf_reg': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5], 'max_features': ['auto', 'sqrt']},
        'xgb_clf': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]},
        'xgb_reg': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]}
    }
    return param_grids

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

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, problem_type):
    """Train and tune models"""
    param_grids = get_param_grids()
    results = []

    if problem_type == 'classification':
        models = {
            'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), param_grids['logistic'], X_train_scaled, X_test_scaled),
            'KNN Classifier': (KNeighborsClassifier(), param_grids['knn_clf'], X_train_scaled, X_test_scaled),
            'SVM Classifier': (SVC(random_state=42), param_grids['svm_clf'], X_train_scaled, X_test_scaled),
            'Decision Tree': (DecisionTreeClassifier(random_state=42), param_grids['dt_clf'], X_train, X_test),
            'Random Forest': (RandomForestClassifier(random_state=42), param_grids['rf_clf'], X_train, X_test),
            'XGBoost': (xgb.XGBClassifier(random_state=42), param_grids['xgb_clf'], X_train, X_test)
        }
        scoring = 'accuracy'
    else:  # regression
        models = {
            'Ridge Regression': (Ridge(random_state=42), param_grids['ridge'], X_train_scaled, X_test_scaled),
            'Lasso Regression': (Lasso(random_state=42), param_grids['lasso'], X_train_scaled, X_test_scaled),
            'KNN Regressor': (KNeighborsRegressor(), param_grids['knn_reg'], X_train_scaled, X_test_scaled),
            'SVM Regressor': (SVR(), param_grids['svm_reg'], X_train_scaled, X_test_scaled),
            'Decision Tree': (DecisionTreeRegressor(random_state=42), param_grids['dt_reg'], X_train, X_test),
            'Random Forest': (RandomForestRegressor(random_state=42), param_grids['rf_reg'], X_train, X_test),
            'XGBoost': (xgb.XGBRegressor(random_state=42), param_grids['xgb_reg'], X_train, X_test)
        }
        scoring = 'r2'

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, (model, param_grid, X_tr, X_te)) in enumerate(models.items()):
        status_text.text(f'Training {name}...')

        try:
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring, n_jobs=-1
            )

            # Fit on appropriate data (scaled or unscaled)
            y_tr = y_train
            y_te = y_test

            grid_search.fit(X_tr, y_tr)

            # Make predictions
            y_pred = grid_search.predict(X_te)

            # Calculate metrics
            if problem_type == 'classification':
                score = accuracy_score(y_te, y_pred)
                cv_score = cross_val_score(grid_search.best_estimator_, X_tr, y_tr, cv=5, scoring='accuracy').mean()
            else:
                score = r2_score(y_te, y_pred)
                cv_score = cross_val_score(grid_search.best_estimator_, X_tr, y_tr, cv=5, scoring='r2').mean()

            results.append({
                'Model': name,
                'Best Score': score,
                'CV Score': cv_score,
                'Best Parameters': grid_search.best_params_,
                'Model Object': grid_search.best_estimator_,
                'Predictions': y_pred,
                'X_test': X_te,
                'y_test': y_te
            })

        except Exception as e:
            st.warning(f"Error training {name}: {e}")

        progress_bar.progress((i + 1) / len(models))

    status_text.text('All models trained successfully!')
    return results

def main():
    st.title("🤖 Machine Learning Hyperparameter Tuning App")
    st.markdown("Upload your dataset and let's find the best model with optimal parameters!")

    df = load_data()

    if df is not None:
        explore_data(df)

        # Select target variable
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

        # Prepare data and train models
        if st.button("🚀 Start Model Training", type="primary"):
            with st.spinner("Preparing data..."):
                X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df, target_col, test_size)

            st.success("Data prepared successfully!")
            st.info(f"🔹 Training set size: {X_train.shape[0]} samples")
            st.info(f"🔹 Test set size: {X_test.shape[0]} samples")

            # Train models
            with st.spinner("Training models... This may take a few minutes."):
                results = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, problem_type)

            if results:
                st.session_state['results'] = results
                st.session_state['problem_type'] = problem_type
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['X_train_scaled'] = X_train_scaled
                st.session_state['X_test_scaled'] = X_test_scaled

                st.success("🎉 All models trained successfully!")

if __name__ == "__main__":
    main()
