# 🎵 TuneSight: ML Hyperparameter Tuning Dashboard

TuneSight is an interactive, user-friendly web application for exploring, visualizing, and auto tuning machine learning models. Built with Streamlit, it empowers users to upload datasets, and visualize data for a variety of ML models—all from an intuitive dashboard.

---

## 🚀 Features
- **Upload CSV/XLSX datasets**
- **Automatic data exploration & visualization**
- **Supports classification, regression, and clustering**
- **Interactive hyperparameter tuning (Grid Search)**
- **Model comparison dashboard**
- **Download trained models**
- **Advanced visualizations (correlation heatmaps, loss curves, etc.)**

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd TuneSight
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💡 Usage

1. **Run the app:**
   ```bash
   streamlit run tunesight.py
   ```
2. **Open your browser:**
   - Go to the local URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501))
3. **Upload your dataset:**
   - Supported formats: `.csv`, `.xlsx`
4. **Explore, visualize, and tune!**

---

## 📦 Supported Models
- Ridge & Lasso Regression
- Logistic Regression
- K-Nearest Neighbors (Classifier & Regressor)
- Support Vector Machines (Classifier & Regressor)
- Decision Trees (Classifier & Regressor)
- Random Forests (Classifier & Regressor)
- XGBoost (Classifier & Regressor)
- Clustering: KMeans, DBSCAN, Agglomerative

---

## 📊 Visualizations
- Correlation heatmaps
- Feature distributions
- Model performance metrics
- Confusion matrices, ROC curves
- Loss curves, tree visualizations

---

## 📝 Notes
- **No coding required!** All operations are performed via the web UI.
- **Data privacy:** All processing is local; your data never leaves your machine.
- **Best viewed on desktop browsers.**

