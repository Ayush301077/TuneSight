import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, mean_squared_error, r2_score, silhouette_score)
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import xgboost as xgb
from io import BytesIO
import base64

# Set page config
st.set_page_config(page_title="ML Hyperparameter Tuning", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .model-result {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

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

def create_visualizations(df, target_col, problem_type):
    """Create data visualizations"""
    st.subheader("📈 Data Visualizations")
    
    # Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        plt.close()
    
    # Data distribution
    st.subheader("Data Distribution")
    for col in numerical_cols[:6]:  # Limit to 6 columns for performance
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
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 and len(np.unique(y)) < 20 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders

def get_param_grids():
    """Define parameter grids for hyperparameter tuning"""
    param_grids = {
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'logistic': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'knn_clf': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'knn_reg': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'svm_clf': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'svm_reg': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'dt_clf': {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'dt_reg': {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['mse', 'mae']
        },
        'rf_clf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5],
            'max_features': ['auto', 'sqrt']
        },
        'rf_reg': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5],
            'max_features': ['auto', 'sqrt']
        },
        'xgb_clf': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'xgb_reg': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
    return param_grids

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

def display_results(results, problem_type):
    """Display model results"""
    st.subheader("🏆 Model Performance Results")
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'Model': r['Model'],
            'Test Score': f"{r['Best Score']:.4f}",
            'CV Score': f"{r['CV Score']:.4f}",
            'Best Parameters': str(r['Best Parameters'])
        }
        for r in results
    ])
    
    # Sort by test score
    results_df = results_df.sort_values('Test Score', ascending=False)
    
    # Display results table
    st.dataframe(results_df, use_container_width=True)
    
    # Display individual model cards
    for result in sorted(results, key=lambda x: x['Best Score'], reverse=True):
        with st.expander(f"📊 {result['Model']} - Score: {result['Best Score']:.4f}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Test Score:**", f"{result['Best Score']:.4f}")
                st.write("**CV Score:**", f"{result['CV Score']:.4f}")
            with col2:
                st.write("**Best Parameters:**")
                for param, value in result['Best Parameters'].items():
                    st.write(f"- {param}: {value}")

def create_model_visualizations(selected_model, problem_type, X_test, y_test, y_pred):
    """Create visualizations for the selected model"""
    st.subheader(f"📊 Visualizations for {selected_model['Model']}")
    
    if problem_type == 'classification':
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            st.subheader("ROC Curve")
            try:
                y_pred_proba = selected_model['Model Object'].predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close()
            except:
                st.info("ROC curve not available for this model")
    
    else:  # regression
        # Predicted vs Actual
        st.subheader("Predicted vs Actual Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        st.pyplot(fig)
        plt.close()
        
        # Residuals plot
        st.subheader("Residuals Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        st.pyplot(fig)
        plt.close()
    
    # Feature importance (if available)
    if hasattr(selected_model['Model Object'], 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_names = [f'Feature_{i}' for i in range(len(selected_model['Model Object'].feature_importances_))]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': selected_model['Model Object'].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        plt.close()

def run_clustering(X_scaled, df):
    """Run clustering algorithms"""
    st.subheader("🎯 Clustering Analysis")
    
    clustering_results = {}
    
    # K-Means
    st.write("**K-Means Clustering**")
    k_range = range(2, min(11, len(X_scaled)))
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    # Find best k
    best_k = k_range[np.argmax(silhouette_scores)]
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_best.fit_predict(X_scaled)
    
    clustering_results['K-Means'] = {
        'labels': kmeans_labels,
        'n_clusters': best_k,
        'silhouette_score': max(silhouette_scores)
    }
    
    st.write(f"Best K: {best_k}, Silhouette Score: {max(silhouette_scores):.3f}")
    
    # Visualize K-Means results
    if X_scaled.shape[1] >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
        ax1.set_title('K-Means Clustering Results')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax1)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'bo-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
        ax2.legend()
        
        st.pyplot(fig)
        plt.close()
    
    return clustering_results


def create_decision_tree_visualization(model, feature_names=None):
    """Create decision tree visualization"""
    if hasattr(model, 'tree_'):
        st.subheader("🌳 Decision Tree Visualization")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, ax=ax, feature_names=feature_names, 
                 filled=True, rounded=True, fontsize=10, max_depth=3)
        ax.set_title("Decision Tree Structure (Max Depth: 3)")
        st.pyplot(fig)
        plt.close()

def create_loss_curves(model, X_train, y_train, X_test, y_test, problem_type):
    """Create loss curves for applicable models"""
    if hasattr(model, 'loss_curve_') or 'XGB' in str(type(model)):
        st.subheader("📈 Loss vs Epochs")
        
        # For XGBoost, we need to create validation curves
        if 'XGB' in str(type(model)):
            # Create validation curve for XGBoost
            train_scores = []
            test_scores = []
            
            # Get the number of estimators
            n_estimators = model.n_estimators
            estimator_range = list(range(10, min(n_estimators + 1, 201), 10))
            
            for n_est in estimator_range:
                temp_model = type(model)(n_estimators=n_est, random_state=42)
                temp_model.fit(X_train, y_train)
                
                train_pred = temp_model.predict(X_train)
                test_pred = temp_model.predict(X_test)
                
                if problem_type == 'classification':
                    train_score = accuracy_score(y_train, train_pred)
                    test_score = accuracy_score(y_test, test_pred)
                else:
                    train_score = r2_score(y_train, train_pred)
                    test_score = r2_score(y_test, test_pred)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            # Plot the curves
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(estimator_range, train_scores, 'b-', label='Training Score')
            ax.plot(estimator_range, test_scores, 'r-', label='Validation Score')
            ax.set_xlabel('Number of Estimators')
            ax.set_ylabel('Score')
            ax.set_title('Learning Curves')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close()

def create_advanced_clustering_analysis(X_scaled):
    """Enhanced clustering analysis with multiple algorithms"""
    st.subheader("🎯 Advanced Clustering Analysis")
    
    clustering_methods = {}
    
    # K-Means with elbow method
    st.write("**K-Means Clustering with Elbow Method**")
    k_range = range(2, min(11, len(X_scaled) // 2))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    st.pyplot(fig)
    plt.close()
    
    # Best k based on silhouette score
    best_k = k_range[np.argmax(silhouette_scores)]
    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = best_kmeans.fit_predict(X_scaled)
    
    clustering_methods['K-Means'] = {
        'labels': kmeans_labels,
        'n_clusters': best_k,
        'silhouette_score': max(silhouette_scores)
    }
    
    # DBSCAN
    st.write("**DBSCAN Clustering**")
    eps_range = np.arange(0.1, 2.0, 0.1)
    min_samples_range = [3, 5, 7, 10]
    
    best_dbscan_score = -1
    best_dbscan_params = {}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:  # More than just noise
                try:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
                        clustering_methods['DBSCAN'] = {
                            'labels': labels,
                            'params': best_dbscan_params,
                            'silhouette_score': score
                        }
                except:
                    continue
    
    # Hierarchical Clustering
    st.write("**Hierarchical Clustering**")
    linkage_methods = ['ward', 'complete', 'average']
    best_hierarchical_score = -1
    
    for linkage in linkage_methods:
        for n_clusters in k_range:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = hierarchical.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_hierarchical_score:
                best_hierarchical_score = score
                clustering_methods['Hierarchical'] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'linkage': linkage,
                    'silhouette_score': score
                }
    
    # Display clustering results
    st.subheader("📊 Clustering Results Comparison")
    comparison_df = pd.DataFrame([
        {
            'Method': method,
            'Silhouette Score': results['silhouette_score'],
            'Parameters': str({k: v for k, v in results.items() if k not in ['labels', 'silhouette_score']})
        }
        for method, results in clustering_methods.items()
    ]).sort_values('Silhouette Score', ascending=False)
    
    st.dataframe(comparison_df)
    
    # Visualize best clustering result
    best_method = comparison_df.iloc[0]['Method']
    best_labels = clustering_methods[best_method]['labels']
    
    if X_scaled.shape[1] >= 2:
        st.subheader(f"🎯 Best Clustering Result: {best_method}")
        
        # Create 2D visualization using first two principal components
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'{best_method} Clustering Results (PCA Projection)')
        plt.colorbar(scatter)
        st.pyplot(fig)
        plt.close()
    
    return clustering_methods

def create_model_comparison_dashboard(results, problem_type):
    """Create an interactive model comparison dashboard"""
    st.subheader("📊 Model Comparison Dashboard")
    
    # Extract metrics for comparison
    model_names = [r['Model'] for r in results]
    test_scores = [r['Best Score'] for r in results]
    cv_scores = [r['CV Score'] for r in results]
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for test scores
        fig_test = px.bar(
            x=model_names, y=test_scores,
            title="Test Score Comparison",
            labels={'x': 'Models', 'y': 'Test Score'},
            color=test_scores,
            color_continuous_scale='viridis'
        )
        fig_test.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_test, use_container_width=True)
    
    with col2:
        # Bar chart for CV scores
        fig_cv = px.bar(
            x=model_names, y=cv_scores,
            title="Cross-Validation Score Comparison",
            labels={'x': 'Models', 'y': 'CV Score'},
            color=cv_scores,
            color_continuous_scale='plasma'
        )
        fig_cv.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # Scatter plot: Test vs CV scores
    fig_scatter = px.scatter(
        x=test_scores, y=cv_scores,
        text=model_names,
        title="Test Score vs Cross-Validation Score",
        labels={'x': 'Test Score', 'y': 'CV Score'},
        size=[1]*len(model_names),
        size_max=15
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.add_shape(
        type="line", line=dict(dash="dash"),
        x0=min(test_scores), y0=min(test_scores),
        x1=max(test_scores), y1=max(test_scores)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def create_model_download(selected_model):
    """Create downloadable model file"""
    try:
        # Create a model package with all necessary components
        model_package = {
            'model': selected_model['Model Object'],
            'model_name': selected_model['Model'],
            'best_parameters': selected_model['Best Parameters'],
            'test_score': selected_model['Best Score'],
            'cv_score': selected_model['CV Score'],
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create a BytesIO object to store the serialized model
        model_buffer = BytesIO()
        
        # Serialize the model using joblib (better for scikit-learn models)
        joblib.dump(model_package, model_buffer)
        
        # Get the bytes
        model_bytes = model_buffer.getvalue()
        
        return model_bytes, f"{selected_model['Model'].replace(' ', '_').lower()}_model.pkl"
    
    except Exception as e:
        st.error(f"Error creating model download: {e}")
        return None, None

def create_prediction_ui():
    """Create UI for making predictions on new data."""
    st.subheader("🔮 Make a Prediction")

    if 'results' not in st.session_state:
        st.info("Train a model first to make predictions.")
        return

    # Get required items from session state
    results = st.session_state['results']
    original_df = st.session_state['original_df_for_inputs']
    feature_columns = st.session_state['feature_columns']
    label_encoders = st.session_state['label_encoders']
    scaler = st.session_state['scaler']
    
    # Get the currently selected model from the other selectbox
    selected_model_name = st.session_state.get('model_selector')
    if not selected_model_name:
         # Fallback to the best model if selector is not available yet
         selected_model_name = sorted(results, key=lambda x: x['Best Score'], reverse=True)[0]['Model']

    selected_model = next(r for r in results if r['Model'] == selected_model_name)

    st.write(f"Enter feature values to predict using **{selected_model_name}**:")

    with st.form("prediction_form"):
        user_inputs = {}
        # Create input fields for each feature
        for col in feature_columns:
            if col in label_encoders:
                # Categorical feature
                options = list(original_df[col].unique())
                user_inputs[col] = st.selectbox(f"Select {col}", options=options, key=f"pred_{col}")
            elif pd.api.types.is_numeric_dtype(original_df[col]):
                # Numerical feature
                user_inputs[col] = st.number_input(f"Enter {col}", value=float(original_df[col].mean()), key=f"pred_{col}")
            else:
                 # Fallback for other types
                user_inputs[col] = st.text_input(f"Enter {col}", value="", key=f"pred_{col}")

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            # Create a dataframe from user input
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[feature_columns] # Ensure correct order

            # Preprocess the input data
            processed_df = input_df.copy()

            # 1. Label Encode categorical features
            for col, encoder in label_encoders.items():
                # The selectbox ensures the value is in the encoder's classes
                processed_df[col] = encoder.transform(processed_df[col])
            
            # 2. Determine if scaling is needed
            scaled_models = [
                'Logistic Regression', 'KNN Classifier', 'SVM Classifier', 
                'Ridge Regression', 'Lasso Regression', 'KNN Regressor', 'SVM Regressor'
            ]
            
            prediction_input = processed_df
            if selected_model_name in scaled_models:
                prediction_input = scaler.transform(processed_df)

            # Make prediction
            prediction = selected_model['Model Object'].predict(prediction_input)
            prediction_proba = None
            if hasattr(selected_model['Model Object'], "predict_proba"):
                try:
                    prediction_proba = selected_model['Model Object'].predict_proba(prediction_input)
                except Exception as e:
                    st.warning(f"Could not get prediction probabilities: {e}")

            # Display prediction
            st.success(f"**Predicted Value:** `{prediction[0]}`")
            if prediction_proba is not None:
                st.write("**Prediction Probabilities:**")
                if hasattr(selected_model['Model Object'], 'classes_'):
                    st.write(pd.DataFrame(prediction_proba, columns=selected_model['Model Object'].classes_))
                else: # For regressors that might have this method by mistake
                    st.write(prediction_proba)

def main():
    st.title("🤖 TuneSight - Tune Precisely, See Clearly")
    st.markdown("Upload your cleaned dataset from DeMessify and find the best model with optimal parameters!")
    st.markdown("---")
    st.markdown("### 🎯 App Features")
    st.markdown("""
    - **Automated Hyperparameter Tuning** for 10+ ML algorithms
    - **Cross-Validation** with customizable folds
    - **Comprehensive Visualizations** for all model types
    - **Interactive Model Comparison** dashboard
    - **Advanced Clustering Analysis** with multiple algorithms
    - **Prediction UI** for new data inputs
    - **Downloadable Results** in CSV format
    """)
    
    st.markdown("---")
    st.markdown("💡 **Tip**: Upload your cleaned dataset and let the app automatically find the best model with optimal parameters!")


    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("### 🔧 App Sections")
    st.sidebar.info("1. 📊 Data Upload & Exploration\n2. 🎯 Target Selection\n3. 🚀 Model Training\n4. 📈 Results Analysis\n5. 🔬 Advanced Analytics")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Data exploration
        explore_data(df)
        
        # Select target variable
        st.subheader("🎯 Target Variable Selection")
        target_col = st.selectbox("Select target variable:", df.columns.tolist())
        
        # Determine problem type
        unique_values = df[target_col].nunique()
        if unique_values <= 10 and df[target_col].dtype in ['object', 'int64']:
            problem_type = st.selectbox("Problem Type:", ['classification', 'regression'])
        else:
            problem_type = 'regression'
        
        st.info(f"Detected problem type: **{problem_type}**")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
            random_state = st.number_input("Random state:", 0, 100, 42)
        
        # Create visualizations
        create_visualizations(df, target_col, problem_type)
        
        # Prepare data and train models
        if st.button("🚀 Start Model Training", type="primary"):
            with st.spinner("Preparing data..."):
                X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoders = prepare_data(df, target_col, test_size)
            
            st.success("Data prepared successfully!")
            st.info(f"Training set size: {X_train.shape[0]} samples")
            st.info(f"Test set size: {X_test.shape[0]} samples")
            
            # Train models
            with st.spinner("Training models... This may take a few minutes."):
                results = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, problem_type)
            
            if results:
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['problem_type'] = problem_type
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['X_train_scaled'] = X_train_scaled
                st.session_state['X_test_scaled'] = X_test_scaled
                st.session_state['scaler'] = scaler
                st.session_state['label_encoders'] = label_encoders
                st.session_state['feature_columns'] = X_train.columns.tolist()
                st.session_state['original_df_for_inputs'] = df
                st.session_state['target_col'] = target_col
                
                st.success("🎉 All models trained successfully!")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        problem_type = st.session_state['problem_type']
        
        # Display results
        display_results(results, problem_type)
        
        # Model comparison dashboard
        create_model_comparison_dashboard(results, problem_type)
        
        # Model selection for detailed analysis
        st.subheader("🎯 Select Model for Detailed Analysis")
        model_names = [r['Model'] for r in results]
        selected_model_name = st.selectbox("Choose a model:", model_names, key='model_selector')
        selected_model = next(r for r in results if r['Model'] == selected_model_name)
        
        # Add prediction section
        create_prediction_ui()
        
        # Create visualizations for selected model
        create_model_visualizations(
            selected_model, problem_type, 
            selected_model['X_test'], selected_model['y_test'], selected_model['Predictions']
        )
        
        # Decision tree visualization
        if 'Tree' in selected_model_name:
            feature_names = [f'Feature_{i}' for i in range(selected_model['X_test'].shape[1])]
            create_decision_tree_visualization(selected_model['Model Object'], feature_names)
        
        # Loss curves for applicable models
        if 'XGB' in selected_model_name or 'Random Forest' in selected_model_name:
            create_loss_curves(
                selected_model['Model Object'], 
                st.session_state['X_train'], st.session_state['y_train'],
                st.session_state['X_test'], st.session_state['y_test'],
                problem_type
            )
        
        # Download section
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download model results
            results_summary = pd.DataFrame([
                {
                    'Model': r['Model'],
                    'Test_Score': r['Best Score'],
                    'CV_Score': r['CV Score'],
                    'Best_Parameters': str(r['Best Parameters'])
                }
                for r in results
            ])
            
            csv = results_summary.to_csv(index=False)
            st.download_button(
                label="📊 Download Results CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
        with col2:
         # Download selected model
            if 'selected_model' in locals():
                model_bytes, filename = create_model_download(selected_model)
        
                if model_bytes is not None:
                    st.download_button(
                        label="🤖 Download Selected Model",
                        data=model_bytes,
                        file_name=filename,
                        mime="application/octet-stream",
                        help="Download the trained model with parameters for future use"
                    )
        
    # Advanced Analysis Section
    if df is not None:
        st.subheader("🔬 Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Run Clustering Analysis"):
                # Prepare data for clustering
                if 'target_col' in locals():
                    X = df.drop(columns=[target_col])
                else:
                    X = df.select_dtypes(include=[np.number])
                
                categorical_cols = X.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                
                X = X.fillna(X.mean())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Run advanced clustering
                clustering_results = create_advanced_clustering_analysis(X_scaled)
                st.session_state['clustering_results'] = clustering_results
        
        
    
if __name__ == "__main__":
    main() 