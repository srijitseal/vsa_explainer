# code generates the SHAP plots (both beeswarm and the mean-abs shap)
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_PATH = "B3DB_regression.tsv" 
SMILES_COL = "SMILES"  
TARGET_COL = "logBB"      
RANDOM_STATE = 142

# --- Load dataset ---
df = pd.read_csv(DATA_PATH, sep = '\t')
df = df.dropna(subset=[SMILES_COL, TARGET_COL])

# Convert SMILES to RDKit molecules
df["mol"] = df[SMILES_COL].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()].reset_index(drop=True)

# --- Compute all RDKit descriptors ---
descriptor_names = [d[0] for d in Descriptors.descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def compute_descriptors(mol):
    return calc.CalcDescriptors(mol)

print(f"Calculating {len(descriptor_names)} descriptors for {len(df)} molecules...")
X = pd.DataFrame([compute_descriptors(m) for m in df["mol"]], columns=descriptor_names)


X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = df[TARGET_COL].astype(float)

print("Feature matrix shape:", X.shape)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# --- Train our Random Forest model---
rf = RandomForestRegressor(n_estimators=800, random_state=RANDOM_STATE, n_jobs=-1,
                           max_depth = 20)
rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_test)
print("Test RMSE:", mean_squared_error(y_test, y_pred))
print("Test R^2:", r2_score(y_test, y_pred))

# --- SHAP explanations ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X_test, show=True)

# SHAP bar plot (mean abs shap values)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

#---------

# The code below picks one sample from the test set to visualize (local visualizations)
i = 5  # index of the sample you want to explain, this says that the 6th compound (0-index reference) is selected

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value,
        data=X.iloc[i],
        feature_names=X.columns
    )
)


#-------
# This code generates the parity plot of the test set (actual vs predicted)

# --- Parity plot ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")


lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Ideal")


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
plt.text(0.05, 0.95,
         f"RMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}",
         transform=plt.gca().transAxes,
         verticalalignment="top",
         fontsize=12,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.xlabel("Experimental logBB")
plt.ylabel("Predicted logBB")
plt.title("Parity Plot (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()
