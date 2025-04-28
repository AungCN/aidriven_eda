# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(page_title="AI-Driven EDA + Automated ML", layout="wide")

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ML Models
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, r2_score, ConfusionMatrixDisplay

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor, plot_importance
    xgb_installed = True
except:
    xgb_installed = False

# Title
st.title("🤖 AI-Driven EDA + Automated Machine Learning")
st.markdown("Upload your dataset, explore EDA, and train multiple ML models with cross-validation and tuning!")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # EDA
    st.subheader("📊 Exploratory Data Analysis")
    st.write(df.head())

    if st.checkbox("Show Shape"):
        st.write(f"Dataset Shape: {df.shape}")

    if st.checkbox("Show Summary Stats"):
        st.write(df.describe())

    if st.checkbox("Show Missing Values"):
        missing = df.isnull().sum()
        st.write(missing[missing > 0])

    if st.checkbox("Correlation Heatmap (Numerical Only)"):
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # Target Variable
    st.sidebar.header("🏷️ Define Target Variable")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Problem Type
    problem_type = st.sidebar.radio("Problem Type", ("Classification", "Regression"))

    # Feature Selection
    features = st.sidebar.multiselect(
        "Select Feature Columns", [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col]
    )

    auto_feature_select = st.sidebar.checkbox("✨ Auto Select Top Features", value=False)
    if auto_feature_select:
        num_features = st.sidebar.slider("Number of Top Features", 1, len(features), min(10, len(features)))

    # Preprocessing
    X = df[features].copy()
    y = df[target_col]

    # 🔥 Missing Value Imputation for X
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    # 🔥 Missing Value Imputation for y
    if y.isnull().sum() > 0:
        if problem_type == "Regression":
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode()[0])

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if auto_feature_select:
        selector = SelectKBest(score_func=f_classif if problem_type=="Classification" else f_regression, k=num_features)
        X_scaled = selector.fit_transform(X_scaled, y)

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)

    # Model Selection
    st.sidebar.header("⚙️ Model Selection")

    available_models = {}

    if problem_type == "Classification":
        available_models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }
        if xgb_installed:
            available_models['XGBoost Classifier'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        available_models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor()
        }
        if xgb_installed:
            available_models['XGBoost Regressor'] = XGBRegressor()

    use_cv = st.sidebar.checkbox("🏋️ Use 5-Fold Cross-Validation", value=True)

    st.sidebar.header(":gear: Hyperparameter Tuning")
    rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 500, 100, step=10)
    if xgb_installed:
        xgb_learning_rate = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.5, 0.1, step=0.01)

    train_all = st.sidebar.checkbox("🚀 Train All Models", value=True)

    selected_models = []
    for model_name in available_models.keys():
        if train_all or st.sidebar.checkbox(model_name, value=True):
            selected_models.append(model_name)

    if st.sidebar.button("Train Selected Models"):
        leaderboard = []

        for model_name in selected_models:
            model = available_models[model_name]

            if 'Random Forest' in model_name:
                model.set_params(n_estimators=rf_n_estimators)
            if 'XGBoost' in model_name and xgb_installed:
                model.set_params(learning_rate=xgb_learning_rate)

            if use_cv:
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == "Classification" else KFold(n_splits=5, shuffle=True, random_state=42)
                if problem_type == "Classification":
                    scores = cross_val_score(model, X_scaled, y, scoring='accuracy', cv=kfold)
                else:
                    scores = cross_val_score(model, X_scaled, y, scoring='neg_root_mean_squared_error', cv=kfold)
                score = np.mean(scores)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if problem_type == "Classification":
                    score = accuracy_score(y_test, preds)
                else:
                    score = np.sqrt(mean_squared_error(y_test, preds))

            leaderboard.append({"Model": model_name, "Score": score})

        leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="Score", ascending=(problem_type=="Regression"))

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(leaderboard_df.style.format({"Score": "{:.4f}"}))

        csv = leaderboard_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Leaderboard CSV", csv, "model_leaderboard.csv", "text/csv")

        best_model_name = leaderboard_df.iloc[0]['Model']
        st.success(f"🌟 Best Model: **{best_model_name}**")

        best_model = available_models[best_model_name]
        if 'Random Forest' in best_model_name:
            best_model.set_params(n_estimators=rf_n_estimators)
        if 'XGBoost' in best_model_name and xgb_installed:
            best_model.set_params(learning_rate=xgb_learning_rate)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        if problem_type == "Classification":
            st.subheader("🔢 Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("🔧 Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.subheader("📈 Regression Metrics")
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            st.write(f"**RMSE:** {rmse:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

        if hasattr(best_model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            try:
                importances = best_model.feature_importances_
                if auto_feature_select:
                    feat_df = pd.DataFrame({'Feature': [f"Feature {i+1}" for i in range(len(importances))], 'Importance': importances})
                else:
                    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Feature importance plotting failed: {e}")

#END
