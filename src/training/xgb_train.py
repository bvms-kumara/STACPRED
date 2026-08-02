import pandas as pd
import numpy as np
import warnings
import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore")

# --- 0. REPORTING FUNCTION ---
def save_report_to_file(critical_data_dict, filename_prefix="model_training_report"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{'MODEL TRAINING AND EVALUATION REPORT':^80}\n")
        f.write(f"{'Generated on: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}\n")
        f.write("="*80 + "\n\n")

        for title, content in critical_data_dict.items():
            f.write("-" * 40 + "\n")
            f.write(f"| {title.upper()} |\n")
            f.write("-" * 40 + "\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    # Handle numpy types in reporting
                    if isinstance(value, (int, float, np.float64, np.float32)):
                        f.write(f"{key:<35}: {value:.4f}\n")
                    elif isinstance(value, list):
                        # Write list elements on one line for CV RMSE
                        list_str = ", ".join([f"{x:.4f}" for x in value]) if len(value) < 10 else f"[List of {len(value)} values]"
                        f.write(f"{key:<35}: [{list_str}]\n")
                    else:
                        f.write(f"{key:<35}: {value}\n")
            else:
                f.write(f"{content}\n")
            f.write("\n")
    print(f"\n[INFO] Report saved to: {filename}")

# --- 1. CONFIG (Set base for High R2 training) ---
CSV_FILES = [
    'b2_ad4_final_results_2_clean.csv',
    'b2_dock6_final_results_2_clean.csv',
    'b2_vina_final_results_2_clean.csv',
    'b2_vinardo_final_results_2_clean.csv'
]
TARGET_COLUMN = 'Binding_Energy'
RANDOM_SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
N_JOBS = -1

# --- 2. LOOP OVER CSV FILES ---
for file_path in CSV_FILES:
    print(f"\n=== Processing: {file_path} ===")
    REPORT_DATA = {}
    REPORT_DATA['CONFIGURATION'] = {
        'File Path': file_path,
        'Target Column': TARGET_COLUMN,
        'Random Seed': RANDOM_SEED,
        'Test Size': TEST_SIZE,
        'Validation Size': VALIDATION_SIZE,
        'N_Jobs': N_JOBS
    }

    # --- 2a. LOAD DATA ---
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    df.drop(columns=['Full_Name', 'Canonical_SMILES'], inplace=True, errors='ignore')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    final_shape = df.shape
    REPORT_DATA['DATASET_INFO'] = {
        'Initial Rows/Columns': initial_shape,
        'After Drop NA': final_shape
    }

    # --- 2b. PREPARE FEATURES & TARGET ---
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # --- 2c. DATA SPLIT ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VALIDATION_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_SEED
    )
    REPORT_DATA['DATA_SPLIT_SIZES'] = {
        'Training': X_train.shape[0],
        'Validation': X_val.shape[0],
        'Test': X_test.shape[0]
    }

    # --- 2d. PREPROCESSING (Switch to MinMaxScaler for potential R2 boost) ---
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        n_jobs=N_JOBS,
        verbose_feature_names_out=False
    )

    # --- 2e. MODEL TRAINING (High R2 Hyperparameters) ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=15000, # Significantly increased
        learning_rate=0.008, # Significantly decreased
        max_depth=9, # Increased
        colsample_bytree=0.8, # Slightly decreased
        subsample=0.8, # Slightly decreased
        random_state=RANDOM_SEED,
        early_stopping_rounds=200, # Increased
        eval_metric='rmse',
        enable_categorical=True,
        device='cuda'
    )

    # Preprocess training and validation sets
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    eval_set = [(X_val_proc, y_val)]
    print("Training model with high R2 parameters...")
    xgb_model.fit(X_train_proc, y_train, eval_set=eval_set, verbose=False)

    best_iter = xgb_model.best_iteration
    print(f"Best iteration: {best_iter}")

    # --- 2f. EVALUATION ---
    def evaluate(model, X_proc, y_true, name="Dataset"):
        y_pred = model.predict(X_proc)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return rmse, mae, r2

    X_test_proc = preprocessor.transform(X_test)
    val_metrics = evaluate(xgb_model, X_val_proc, y_val, "Validation")
    test_metrics = evaluate(xgb_model, X_test_proc, y_test, "Test")

    # --- 2g. CROSS-VALIDATION (Adjusted for final training parameters) ---
    print("Performing 5-fold cross-validation...")
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', xgb.XGBRegressor(
                                        objective='reg:squarederror',
                                        n_estimators=best_iter + 500, # Used the best_iter from early stopping
                                        learning_rate=0.008, # Consistent with training
                                        max_depth=9, # Consistent with training
                                        colsample_bytree=0.8,
                                        subsample=0.8,
                                        random_state=RANDOM_SEED,
                                        eval_metric='rmse',
                                        enable_categorical=True,
                                        device='cuda'
                                    ))])
    cv_scores = cross_val_score(full_pipeline, X_train_val, y_train_val,
                                scoring='neg_root_mean_squared_error', cv=5, n_jobs=N_JOBS)
    cv_rmse = -cv_scores
    print(f"CV 5-fold RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

    # --- 2h. FINAL TRAINING & SAVE MODEL ---
    print("Refitting final model on full training+validation set...")
    full_pipeline.fit(X_train_val, y_train_val)
    model_filename = f"{file_path.split('_clean')[0]}_model.pkl"
    joblib.dump(full_pipeline, model_filename)
    print(f"Model saved to: {model_filename}")

    # --- 2i. REPORT ---
    REPORT_DATA['TRAINING_METRICS'] = {
        'Validation RMSE/MAE/R2': val_metrics,
        'Test RMSE/MAE/R2': test_metrics,
        'CV 5-fold RMSE': cv_rmse.tolist(),
        'CV Mean RMSE': float(cv_rmse.mean()),
        'CV Std RMSE': float(cv_rmse.std())
    }
    REPORT_DATA['MODEL_SAVE'] = {'Filename': model_filename}

    save_report_to_file(REPORT_DATA, filename_prefix=file_path.split('_clean')[0]+"_report")

print("\nAll CSVs processed. Models and reports generated successfully.")
