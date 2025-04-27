# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(page_title="AI-Driven EDA + AutoML Pro+", layout="wide")

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
st.title("🤖 AI-Driven EDA + AutoML Pro+")
st.markdown("Upload your dataset, explore EDA, and train multiple ML models with cross-validation and tuning!")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # EDA
    st.subheader("📊 Exploratory Data Analysis")
    st.write(df.head())

    if st.checkbox("Show Shape"):
        st.write(df.shape)

    if st.checkbox("Show Summary Stats"):
        st.write(df.describe())

    if st.checkbox("Show Missing Values"):
        st.write(df.isnull().sum())

    if st.checkbox("Correlation Heatmap (Numerical Only)"):
        corr = df.select_dtypes(include=np.number).corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Target Variable
    st.sidebar.header("🏷️ Define Target Variable")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Problem Type
    problem_type = st.sidebar.radio("Problem Type", ("Classification", "Regression"))

    # Feature Selection
    features = st.sidebar.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col], default=[col for col in df.columns if col != target_col])

    auto_feature_select = st.sidebar.checkbox("✨ Auto Select Top Features", value=False)
    if auto_feature_select:
        num_features = st.sidebar.slider("Number of Top Features", 1, len(features), min(10, len(features)))

    # Preprocessing
    X = df[features]
    y = df[target_col]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
            'Logistic Regression': LogisticRegression(),
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

    # Cross-Validation Option
    use_cv = st.sidebar.checkbox("🏋️ Use 5-Fold Cross-Validation", value=True)

    # Manual Hyperparameter Tuning
    st.sidebar.header(":gear: Hyperparameter Tuning")
    rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 500, 100, step=10)
    if xgb_installed:
        xgb_learning_rate = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.5, 0.1, step=0.01)

    # Train Models
    train_all = st.sidebar.checkbox("🚀 Train All Models", value=True)

    selected_models = []
    for model_name in available_models.keys():
        if train_all or st.sidebar.checkbox(model_name, value=True):
            selected_models.append(model_name)

    # Training
    if st.sidebar.button("Train Selected Models"):
        leaderboard = []

        for model_name in selected_models:
            model = available_models[model_name]

            # Apply tuned parameters
            if 'Random Forest' in model_name:
                model.set_params(n_estimators=rf_n_estimators)
            if 'XGBoost' in model_name and xgb_installed:
                model.set_params(learning_rate=xgb_learning_rate)

            if use_cv:
                if problem_type == "Classification":
                    min_class_count = np.min(np.bincount(y))
                    n_splits = min(5, min_class_count)
                    if n_splits < 2:
                        st.warning("⚠️ Not enough samples for cross-validation. Switching to train/test split only.")
                        use_cv = False
                    else:
                        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                else:
                    n_splits = min(5, len(y))
                    if n_splits < 2:
                        st.warning("⚠️ Not enough samples for cross-validation. Switching to train/test split only.")
                        use_cv = False
                    else:
                        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            if use_cv:
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

        # Show Leaderboard
        st.subheader("🏆 Model Leaderboard")
        st.dataframe(leaderboard_df)

        # Download CSV
        csv = leaderboard_df.to_csv(index=False)
        st.download_button("Download Leaderboard CSV", csv, "model_leaderboard.csv", "text/csv")

        # Best Model Highlight
        best_model_name = leaderboard_df.iloc[0]['Model']
        st.success(f"🌟 Best Model: **{best_model_name}**")

        # Retrain Best Model for Report
        best_model = available_models[best_model_name]
        if 'Random Forest' in best_model_name:
            best_model.set_params(n_estimators=rf_n_estimators)
        if 'XGBoost' in best_model_name and xgb_installed:
            best_model.set_params(learning_rate=xgb_learning_rate)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        if problem_type == "Classification":
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            st.pyplot(plt.gcf())

            st.subheader("🔧 Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        # Feature Importance
        if 'Random Forest' in best_model_name or 'XGBoost' in best_model_name or 'Decision Tree' in best_model_name:
            st.subheader("🔍 Feature Importance")
            try:
                importances = best_model.feature_importances_
                if auto_feature_select:
                    feat_df = pd.DataFrame({'Feature': [f"Feature {i+1}" for i in range(len(importances))], 'Importance': importances})
                else:
                    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False)
                plt.figure(figsize=(10,6))
                sns.barplot(x='Importance', y='Feature', data=feat_df)
                st.pyplot(plt)
            except:
                st.warning("Feature importance not available for this model.")

# END
