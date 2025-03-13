import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import shap
import matplotlib.pyplot as plt

# ---------------------------
# 1. Data Preparation
# ---------------------------
column_names = [
    "Solvent_SMILES",
    "Density",
    "Polymer_SMILES",
    "Interaction_Label",
    "Encoded_Label",
    "N",
    "T",
    "Rg_mean",
    "Rg_SD",
    "SASA_mean",
    "SASA_SD",
    "D_mean",
    "Re_mean",
    "monomer_NumHDonors",
    "monomer_NumHAcceptors",
    "monomer_MolWt",
    "monomer_MolLogP",
    "monomer_MolMR",
    "monomer_TPSA",
    "monomer_NumRotatableBonds",
    "monomer_RingCount",
    "monomer_FractionCSP3",
    "solvent_NumHDonors",
    "solvent_NumHAcceptors",
    "solvent_MolWt",
    "solvent_MolLogP",
    "solvent_MolMR",
    "solvent_TPSA",
    "solvent_NumRotatableBonds",
    "solvent_RingCount",
    "solvent_FractionCSP3",
]

df = pd.read_csv(
    "solvation_prediction/solvent_polymer_predictions.csv",
    header=0,
    names=column_names,
)
# Correct label mapping (original: 0=solvation, 1=no, 2=interaction)
df["Encoded_Label"] = df["Encoded_Label"].map({0: 2, 1: 0, 2: 1})
assert not df["Encoded_Label"].isna().any(), "Label encoding issue."

features = [
    "Rg_mean",
    "Rg_SD",
    "SASA_mean",
    "SASA_SD",
    "D_mean",
    "Re_mean",
    "monomer_NumHDonors",
    "monomer_NumHAcceptors",
    "monomer_MolWt",
    "monomer_MolLogP",
    "monomer_MolMR",
    "monomer_TPSA",
    "monomer_NumRotatableBonds",
    "monomer_RingCount",
    "monomer_FractionCSP3",
    "solvent_NumHDonors",
    "solvent_NumHAcceptors",
    "solvent_MolWt",
    "solvent_MolLogP",
    "solvent_MolMR",
    "solvent_TPSA",
    "solvent_NumRotatableBonds",
    "solvent_RingCount",
    "solvent_FractionCSP3",
]

X = df[features]

# Stage 1 Encoding: Binary event
df["Solvation_Event"] = df["Encoded_Label"].apply(lambda x: 0 if x == 0 else 1)

# Split dataset (Stage 1)
X_train, X_test, y_train_evt, y_test_evt = train_test_split(
    X,
    df["Solvation_Event"],
    test_size=0.2,
    stratify=df["Solvation_Event"],
    random_state=42,
)

# ---------------------------
# 2. Stage 1: Binary XGBoost Model
# ---------------------------
model_evt = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model_evt.fit(X_train, y_train_evt)

# Evaluate clearly Stage 1
y_pred_evt = model_evt.predict(X_test)
print("Stage 1: Solvation Event Classification")
print(
    classification_report(
        y_test_evt, y_pred_evt, target_names=["No Solvation", "Solvation Event"]
    )
)

# Confusion Matrix
cm_evt = confusion_matrix(y_test_evt, y_pred_evt)
ConfusionMatrixDisplay(
    cm_evt, display_labels=["No Solvation", "Solvation Event"]
).plot()
plt.title("Stage 1 Confusion Matrix")
plt.show()

# ---------------------------
# 3. Stage 2: Conditional Model (Interaction vs Full Solvation)
# ---------------------------
df_stage2 = df[df["Solvation_Event"] == 1].copy()
X_stage2 = df_stage2[features]
y_stage2 = df_stage2["Encoded_Label"].map(
    {1: 0, 2: 1}
)  # 0=Interaction, 1=Full Solvation

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_stage2, y_stage2, test_size=0.2, stratify=y_stage2, random_state=42
)

model_type = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model_type.fit(X_train2, y_train2)

# Evaluate Stage 2
y_pred2 = model_type.predict(X_test2)
print("Stage 2: Interaction vs Full Solvation Classification")
print(
    classification_report(
        y_test2, y_pred2, target_names=["Interaction", "Full Solvation"]
    )
)

cm_type = confusion_matrix(y_test2, y_pred2)
ConfusionMatrixDisplay(cm_type, display_labels=["Interaction", "Full Solvation"]).plot()
plt.title("Stage 2 Confusion Matrix")
plt.show()

# ---------------------------
# 4. Combine and Save Results
# ---------------------------
y_test_results = X_test.copy()
y_test_results["P_Solvation_Event"] = model_evt.predict_proba(X_test)[:, 1]
y_test_results["P_Full_Solvation_given_Event"] = model_type.predict_proba(X_test)[:, 1]
y_test_results["P_Interaction_given_Event"] = (
    1 - y_test_results["P_Full_Solvation_given_Event"]
)
y_test_results["Final_Prediction"] = y_test_results.apply(
    lambda row: (
        "No Solvation"
        if row["P_Solvation_Event"] < 0.5
        else (
            "Full Solvation"
            if row["P_Full_Solvation_given_Event"] >= 0.5
            else "Interaction"
        )
    ),
    axis=1,
)
y_test_results.to_csv("final_hierarchical_predictions.csv", index=False)

# ---------------------------
# 5. Complete SHAP Analysis (Stage 1 and Stage 2)
# ---------------------------
shap.initjs()

# Stage 1 SHAP Analysis
explainer_evt = shap.TreeExplainer(model_evt)
shap_evt = explainer_evt.shap_values(X_train)

# Stage 1 SHAP Plots
shap.summary_plot(shap_evt, X_train, plot_type="bar")
shap.summary_plot(shap_evt, X_train)
shap.dependence_plot("Rg_mean", shap_evt, X_train)

# Example Force Plot (Stage 1), this is just for Rg right now, need it for all
########################################################################
shap.force_plot(
    explainer_evt.expected_value, shap_evt[0, :], X_train.iloc[0], matplotlib=True
)

# Decision Plot (Stage 1)
shap.decision_plot(explainer_evt.expected_value, shap_evt[:20], X_train.iloc[:20])

# Stage 2 SHAP Analysis
explainer_type = shap.TreeExplainer(model_type)
shap_type = explainer_type.shap_values(X_train2)

# Stage 2 SHAP Plots
shap.summary_plot(shap_type, X_train2, plot_type="bar")
shap.summary_plot(shap_type, X_train2)
shap.dependence_plot("SASA_mean", shap_type, X_train2)

# Example Force Plot (Stage 2)
shap.force_plot(
    explainer_type.expected_value, shap_type[0, :], X_train2.iloc[0], matplotlib=True
)

# Decision Plot (Stage 2)
shap.decision_plot(explainer_type.expected_value, shap_type[:20], X_train2.iloc[:20])


# Repeat clearly for other key features as desired
# Violin plot for Stage 1 (Solvation Event)
shap.plots.violin(
    shap_values=shap_evt,  # SHAP values for first-stage model
    features=X_train,  # Feature matrix
    feature_names=X_train.columns,  # Feature names
    plot_type="layered_violin",  # Layered violin plot
)


# Ensure SHAP values are computed correctly
shap_values_evt = explainer_evt(X_train)
shap_values_type = explainer_type(X_train2)

# 🎨 Generate SHAP Heatmap (Stage 1 - Solvation Event)
shap.plots.heatmap(shap_values_evt)

# 🎨 Generate SHAP Heatmap (Stage 2 - Interaction vs. Full Solvation)
shap.plots.heatmap(shap_values_type)
