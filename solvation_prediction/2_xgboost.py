import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import numpy as np

# Column names defined
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

# Load your data
df = pd.read_csv(
    "solvation_prediction/solvent_polymer_predictions.csv",
    header=0,
    names=column_names,
)

# Correct label mapping (existing encoding: 0=Solvation, 1=No, 2=Interaction)
correct_label_map = {0: 2, 1: 0, 2: 1}
df["Encoded_Label"] = df["Encoded_Label"].map(correct_label_map)

# Verify encoding correctness
assert not df["Encoded_Label"].isna().any(), "Found unmapped labels!"

# Define features clearly
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

# First binary target: Solvation Event (Interaction or Solvation)
df["Solvation_Event"] = df["Encoded_Label"].apply(lambda x: 0 if x == 0 else 1)

X_train, X_test, y_train_event, y_test_event = train_test_split(
    X,
    df["Solvation_Event"],
    test_size=0.2,
    random_state=42,
    stratify=df["Solvation_Event"],
)

# First-stage model (Solvation event)
model_event = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model_event.fit(X_train, y_train_event)

# Predict solvation event probabilities
y_event_prob = model_event.predict_proba(X_test)[:, 1]
y_event_pred = model_event.predict(X_test)

print("First Stage: Solvation Event (Yes/No)")
print(
    classification_report(
        y_test_event, y_event_pred, target_names=["No Solvation", "Solvation Event"]
    )
)

# SHAP for first-stage model
explainer_event = shap.TreeExplainer(model_event)
shap_values_event = explainer_event.shap_values(X_train)

shap.summary_plot(
    shap_values_event, X_train, plot_type="bar", max_display=X_train.shape[1]
)

# Prepare second-stage data (interaction vs full solvation)
df_event = df[df["Solvation_Event"] == 1].copy()
X_event = df_event[features]
y_event_type = df_event["Encoded_Label"].apply(
    lambda x: 0 if x == 1 else 1
)  # 0=interaction, 1=solvation

X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(
    X_event, y_event_type, test_size=0.2, random_state=42, stratify=y_event_type
)

# Second-stage model (Interaction vs Full Solvation)
model_type = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model_type.fit(X_train_ev, y_train_ev)

# Predict second-stage probabilities
y_type_prob = model_type.predict_proba(X_test)[
    :, 1
]  # Probability of full solvation given event
y_type_pred = model_type.predict(X_test)

# Combine both-stage results
final_results = X_test.copy()
final_results["P_Solvation_Event"] = y_event_prob
final_results["P_Full_Solvation_given_Event"] = y_type_prob
final_results["P_Interaction_given_Event"] = 1 - y_type_prob


# Calculate final clear labels
def interpret_results(row):
    if row["P_Solvation_Event"] < 0.5:
        return "No Solvation"
    elif row["P_Full_Solvation_given_Event"] >= 0.5:
        return "Full Solvation"
    else:
        return "Interaction"


final_results["Final_Prediction"] = final_results.apply(interpret_results, axis=1)

final_results.to_csv("hierarchical_predictions.csv", index=False)

# Evaluate second stage (conditional)
print("Second Stage: Interaction vs Full Solvation (conditional)")
print(
    classification_report(
        y_test_ev,
        model_type.predict(X_test_ev),
        target_names=["Interaction", "Full Solvation"],
    )
)

# SHAP for second-stage model
explainer_type = shap.TreeExplainer(model_type)
shap_values_type = explainer_type.shap_values(X_train_ev)

shap.summary_plot(
    shap_values_type, X_train_ev, plot_type="bar", max_display=X_train_ev.shape[1]
)


X_shap = X_train  # Using your existing X_train from the first stage
explainer = shap.TreeExplainer(model_event)
shap_values = explainer.shap_values(X_shap)

# SHAP bar summary plot (Global Feature Importance)
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=len(features))

# SHAP beeswarm (importance and feature impact distribution)
shap.summary_plot(shap_values, X_shap, max_display=X_shap.shape[1])

# Individual SHAP Dependence plots for top 3 important features
top_features = X_shap.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-3:]]
for feature in top_features:
    shap.dependence_plot(feature, shap_values, X_shap)

# SHAP Force Plot for a single prediction example
shap.initjs()
sample_index = 0  # choose an interesting example
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index, :],
    X_shap.iloc[sample_index],
    matplotlib=True,
)

# SHAP Decision Plot (shows decision-making across dataset)
shap.decision_plot(
    explainer.expected_value,
    shap_values[:50],
    X_shap.iloc[:50],
    feature_display_range=slice(None, -31, -1),
)


###############################
