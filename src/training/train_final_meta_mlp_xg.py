import pandas as pd
import numpy as np
import joblib
import warnings
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TARGET_COLUMN = 'Binding_Energy'
RANDOM_SEED = 42
TEST_SIZE = 0.15

# --- DEFINITION OF ALL 8 BASE MODELS TO STACK ---
# Each tuple: (Prediction Column Name, Model Path, Data File Path)
# NOTE: Using the '_tuned' suffix for MLP models for consistency with the last run.
MODELS_TO_STACK = [
    # 4 XGBoost Models (Original)
    ('AD4_XGB_PRED', 'b2_ad4_final_results_2_model.pkl', 'b2_ad4_final_results_2_clean.csv'),
    ('DOCK6_XGB_PRED', 'b2_dock6_final_results_2_model.pkl', 'b2_dock6_final_results_2_clean.csv'),
    ('VINA_XGB_PRED', 'b2_vina_final_results_2_model.pkl', 'b2_vina_final_results_2_clean.csv'),
    ('VINARDO_XGB_PRED', 'b2_vinardo_final_results_2_model.pkl', 'b2_vinardo_final_results_2_clean.csv'),

    # 4 MLP Models (New Highly Tuned)
    ('AD4_MLP_PRED', 'b2_ad4_final_results_2_mlp_model_tuned.pkl', 'b2_ad4_final_results_2_clean.csv'),
    ('DOCK6_MLP_PRED', 'b2_dock6_final_results_2_mlp_model_tuned.pkl', 'b2_dock6_final_results_2_clean.csv'),
    ('VINA_MLP_PRED', 'b2_vina_final_results_2_mlp_model_tuned.pkl', 'b2_vina_final_results_2_clean.csv'),
    ('VINARDO_MLP_PRED', 'b2_vinardo_final_results_2_mlp_model_tuned.pkl', 'b2_vinardo_final_results_2_clean.csv')
]

# Extract unique data file paths for initial alignment check
DATA_FILES = list(set([item[2] for item in MODELS_TO_STACK]))

COLUMNS_TO_DROP_IN_PREDICTION = ['Full_Name', 'Canonical_SMILES']

# --- 1. ALIGN AND PREPARE ALL DATA ---
print("--- Starting 8-Model Consensus Ensemble Stacking Process ---")
print(f"Phase 1: Aligning {len(DATA_FILES)} unique datasets and preparing ground truth.")

aligned_data_by_path = {}
all_indices = None # This will hold the final common index (rows)
initial_row_counts = {}

# Loop over only the unique data files to find the common intersection
for data_path in DATA_FILES:
    print(f"Loading data from {data_path}...")

    # Load and clean the raw data
    df = pd.read_csv(data_path)
    initial_row_counts[data_path] = len(df)

    # CRITICAL CLEANING: ONLY drop rows with NA in the target variable
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Create the master index: Use both InChIKey and a synthetic pose rank
    df['Pose_Rank'] = df.groupby('Standard_InChIKey').cumcount() + 1
    df['Master_ID'] = df['Standard_InChIKey'] + "_" + df['Pose_Rank'].astype(str)

    # Set Master_ID as index and store data
    df.set_index('Master_ID', inplace=True)
    aligned_data_by_path[data_path] = df

    # Find the intersection of indices across all files
    if all_indices is None:
        all_indices = df.index
    else:
        # Intersect the current index with the running intersection
        all_indices = all_indices.intersection(df.index)

# Check if alignment was successful
print(f"\nInitial row counts before alignment (Target Cleaned): {initial_row_counts}")
print(f"Total aligned rows (Intersection of all 4 datasets): {len(all_indices)}")
if len(all_indices) < 100:
    print("CRITICAL WARNING: Intersection is too small for reliable stacking.")

# --- 2. BUILD META-FEATURE MATRIX (8 COLUMNS) ---
X_meta_features = {}
y_meta_target = None

print("\nPhase 2: Generating 8 Meta-Features (Predictions) from 8 Aligned Models...")

for name, model_path, data_path in MODELS_TO_STACK:
    # 1. Get the correct aligned DataFrame
    df_aligned = aligned_data_by_path[data_path].loc[all_indices]

    try:
        # 2. Load the specific model (XGBoost or MLP)
        model_pipeline = joblib.load(model_path)
        print(f"Loaded {name} model from {model_path}.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_path}. Error: {e}")
        exit()

    # 3. Prepare features for prediction
    X_predict = df_aligned.drop(columns=[TARGET_COLUMN] + COLUMNS_TO_DROP_IN_PREDICTION + ['Pose_Rank'], errors='ignore')

    # Impute missing values with the median (consistent with MLP preprocessor)
    # This must be done here, as the loaded pipeline expects clean input data.
    # We impute here to avoid any issues with missing values in the feature matrix X_predict
    # before it enters the preprocessor within the pipeline.
    X_predict[X_predict.select_dtypes(include=np.number).columns] = X_predict.select_dtypes(include=np.number).fillna(X_predict.select_dtypes(include=np.number).median())
    
    # 4. Generate predictions (the meta-feature)
    y_pred = model_pipeline.predict(X_predict).flatten()
    X_meta_features[name] = y_pred

    # Set the target (y_meta_target) based on the first aligned dataframe
    if y_meta_target is None:
        y_meta_target = df_aligned[TARGET_COLUMN]

# Convert the meta-feature dictionary into the final training data (X_meta, y_meta)
X_meta = pd.DataFrame(X_meta_features, index=all_indices)
y_meta = y_meta_target

print(f"\nMeta-Feature Matrix created. Total rows: {len(X_meta)}")
print(f"Features are: {list(X_meta.columns)}")
print(f"Total {len(X_meta.columns)} base model predictions will be used.")

# --- 3. TRAIN AND EVALUATE META-MODEL ---

if len(X_meta) < 100:
    print("\n--- WARNING: SKIPPING FORMAL EVALUATION DUE TO INSUFFICIENT ALIGNED DATA ---")
    print("The stacked model CANNOT be reliably evaluated with so few data points.")
    exit()

# Split data for Meta-Learner training and testing
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X_meta, y_meta, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# Train Meta-Learner (RidgeCV is highly effective here as it finds the optimal weights)
meta_learner = RidgeCV(alphas=np.logspace(-3, 1, 100))
print("\nPhase 3: Training 8-feature Meta-Learner (RidgeCV) and evaluating...")
meta_learner.fit(X_meta_train, y_meta_train)

# Evaluation
y_pred_stacked = meta_learner.predict(X_meta_test)
rmse_stacked = np.sqrt(mean_squared_error(y_meta_test, y_pred_stacked))
r2_stacked = r2_score(y_meta_test, y_pred_stacked)
mae_stacked = mean_absolute_error(y_meta_test, y_pred_stacked)


# --- 4. REPORT WEIGHTS AND METRICS ---
print("\n" + "="*80)
print("### FINAL CONSENSUS (STACKED) MODEL PERFORMANCE ###")
print("="*80)
print(f"Meta-Learner Target: {TARGET_COLUMN} (Experimental Binding Energy)")
print(f"Meta-Learner Algorithm: Ridge Regressor")
print("-" * 80)

# Report Weights
weights_df = pd.DataFrame({
    'Model': X_meta.columns,
    'Weight (Coefficient)': meta_learner.coef_.flatten()
})
weights_df['Weight (Coefficient)'] = weights_df['Weight (Coefficient)'].apply(lambda x: f"{x:.4f}")
print("Base Model Weights (Coefficients):")
print(weights_df.to_markdown(index=False))

print("-" * 80)
print(f"Final Test RMSE (Stacked Score): {rmse_stacked:.4f}")
print(f"Final Test MAE (Stacked Score): {mae_stacked:.4f}")
print(f"Final Test R2 (Stacked Score): {r2_stacked:.4f}")
print("="*80)

# Save the final stacking model for future predictions
model_filename = "final_consensus_8model_stacking_model.pkl"
joblib.dump(meta_learner, model_filename)
print(f"\nFinal Consensus Model saved to: {model_filename}")
print("The script successfully created the unique, combined prediction model.")
