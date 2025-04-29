import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report,
    confusion_matrix, r2_score, ConfusionMatrixDisplay
)
from sklearn.svm import SVC, SVR

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgb_installed = True
except ImportError:
    lgb_installed = False

st.set_page_config(page_title="AI-Driven EDA + Automated Machine Learning", layout="wide")
st.title("🤖 AI-Driven EDA + AutoML")
st.markdown("Upload your dataset, explore EDA, and train ML models with cross-validation and tuning!")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Exploratory Data Analysis")
    st.write(df.head())

    if st.checkbox("Show Dataset Info"):
        st.write(df.describe())
        st.write("Missing Values:", df.isnull().sum())

    if st.checkbox("Correlation Heatmap"):
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.sidebar.header("1️⃣ Define ML Settings")
    target_col = st.sidebar.selectbox("Target Variable", df.columns)
    problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])

    features = st.sidebar.multiselect("Feature Columns", [col for col in df.columns if col != target_col],
                                      default=[col for col in df.columns if col != target_col])
    auto_feature_select = st.sidebar.checkbox("Auto Feature Selection", value=False)
    if auto_feature_select:
        k_features = st.sidebar.slider("Select Top K Features", 1, len(features), 5)

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

    # Preprocessing
    X = df[features].copy()
    y = df[target_col].copy()

    for col in X.columns:
        if X[col].dtype in ["float64", "int64"]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    if y.isnull().sum() > 0:
        y = y.fillna(y.mode()[0])

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if auto_feature_select:
        score_func = f_classif if problem_type == "Classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k_features)
        X_scaled = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size / 100, random_state=42)

    use_cv = st.sidebar.checkbox("Use 5-Fold Cross-Validation", value=True)
    kfold = None
    if use_cv:
        if problem_type == "Classification":
            y_int = y.astype(int)
            min_class_count = np.min(np.bincount(y_int))
            if min_class_count >= 2:
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                use_cv = False
        else:
            if len(y) >= 5:
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            else:
                use_cv = False

    st.sidebar.header("2️⃣ Select Models")
    models = {}
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True)
        }
        if lgb_installed:
            models["LightGBM"] = LGBMClassifier()
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR()
        }
        if lgb_installed:
            models["LightGBM Regressor"] = LGBMRegressor()

    train_all = st.sidebar.checkbox("Train All Models", value=True)
    selected_models = [m for m in models.keys() if train_all or st.sidebar.checkbox(m, value=False)]

    st.sidebar.header("3️⃣ Hyperparameter Tuning")
    rf_n = st.sidebar.slider("Random Forest n_estimators", 50, 300, 100)

    st.sidebar.header("4️⃣ Train & Evaluate")
    if st.sidebar.button("Train Models"):
        results = []
        for model_name in selected_models:
            model = models[model_name]
            try:
                if "Random Forest" in model_name:
                    model.set_params(n_estimators=rf_n)

                if use_cv:
                    scoring = "accuracy" if problem_type == "Classification" else "neg_root_mean_squared_error"
                    score = cross_val_score(model, X_scaled, y, cv=kfold, scoring=scoring).mean()
                    final_score = score if problem_type == "Classification" else -score
                else:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    final_score = accuracy_score(y_test, preds) if problem_type == "Classification" else np.sqrt(mean_squared_error(y_test, preds))

                results.append({"Model": model_name, "Score": final_score})
                joblib.dump(model, f"{model_name.replace(' ', '_')}.pkl")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {e}")

        leaderboard = pd.DataFrame(results).sort_values(by="Score", ascending=(problem_type == "Regression"))
        st.subheader("🏆 Leaderboard")
        st.dataframe(leaderboard)

        st.download_button("📥 Download Leaderboard", leaderboard.to_csv(index=False), "leaderboard.csv")

        if not leaderboard.empty:
            best_model_name = leaderboard.iloc[0]["Model"]
            st.success(f"Best model: {best_model_name}")

            best_model = joblib.load(f"{best_model_name.replace(' ', '_')}.pkl")
            best_model.fit(X_train, y_train)
            preds = best_model.predict(X_test)

            if problem_type == "Classification":
                st.subheader("📊 Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, preds)
                ConfusionMatrixDisplay(cm).plot(cmap='Blues', ax=ax)
                st.pyplot(fig)

                st.subheader("Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.subheader("📈 Regression Report")
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R² Score: {r2:.4f}")

            if hasattr(best_model, 'feature_importances_'):
                st.subheader("🔍 Feature Importances")
                importances = best_model.feature_importances_
                feat_names = [f"Feature {i+1}" for i in range(len(importances))] if auto_feature_select else features
                fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
                fi_df = fi_df.sort_values(by="Importance", ascending=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
                st.pyplot(fig)

    st.sidebar.header("📤 Load a Saved Model")
    model_file = st.sidebar.file_uploader("Upload Saved Model (.pkl)", type=["pkl"])
    if model_file:
        loaded_model = joblib.load(model_file)
        st.success("Model loaded successfully. Ready for predictions!")
