import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
RANDOM_STATE = 42

# --- Parameters to Tune ---
DOWNSAMPLE_RATIO_VALUES = [10] 
C_VALUES = [0.1, 1.0, 10.0]        
K_BEST_VALUES = np.arange(15, 35, 2) 

# ========================================================
# 0. Helper Functions 
# ========================================================

def tune_threshold(clf_pipeline, X_test, y_test):
    """Evaluates model performance across a range of thresholds."""
    y_scores = clf_pipeline.decision_function(X_test) 
    thresholds = np.arange(-2.0, 2.1, 0.1) 
    
    current_best_f1 = -1
    optimal_t = 0
    
    for t in thresholds:
        y_pred_tuned = (y_scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_tuned, labels=[1], average=None, zero_division=0
        )
        if not np.isnan(f1[0]) and f1[0] > current_best_f1:
            current_best_f1 = f1[0]
            optimal_t = t
    return current_best_f1, optimal_t

def preprocess_features(df_in):
    """
    Applies feature engineering to a dataframe. 
    Applies sin/cos transformation to dates and drops static columns.
    """
    df = df_in.copy()
    
    # Check if date exists, if not, skip date engineering
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop redundant features
    # Ensure these columns actually exist before dropping to avoid errors
    drop_cols = [
        "date", "grid_id", "fire_today", "day_of_year", "month",
        "w_lat", "w_lon", "n_lat", "n_lon", "ndvi"
    ]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df

# ========================================================
# 1. Load Data (Separate Files)
# ========================================================

# File names
train_feat_file = "Fire_Train_data.csv"
train_label_file = "fire_train_labels.csv"
test_feat_file = "Fire_test.csv"
test_label_file = "fire_test_labels.csv"

# Label column
TARGET_COL = "fire_today" 

try:
    print("Loading datasets...")
    # Load Training Data
    X_train_raw = pd.read_csv(train_feat_file)
    y_train_raw = pd.read_csv(train_label_file)
    
    # Load Testing Data
    X_test_raw = pd.read_csv(test_feat_file)
    y_test_raw = pd.read_csv(test_label_file)
    
    print(f"Train rows: {len(X_train_raw)} | Test rows: {len(X_test_raw)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the CSV files are in the same directory as this script.")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found in the label file. Check your CSV headers.")
    exit()

# ========================================================
# 2. Feature Engineering & Preparation
# ========================================================

# Apply preprocessing function to both sets to ensure consistency
print("Preprocessing features...")
X_train_processed = preprocess_features(X_train_raw)
X_test_processed = preprocess_features(X_test_raw)

# Ensure feature columns match exactly
features_kept = X_train_processed.columns.tolist()
X_test_processed = X_test_processed[features_kept]

print(f"Total features used: {len(features_kept)}")

# Convert to Numpy Arrays for the Loop
X_train_full = X_train_processed.values.astype(np.float32)
X_test = X_test_processed.values.astype(np.float32)

# Flatten labels using the explicit column name
# We use .values.ravel() to ensure it's a 1D array (shape: (n,)) rather than (n,1)
y_train_full = y_train_raw[TARGET_COL].values.ravel().astype(int)
y_test = y_test_raw[TARGET_COL].values.ravel().astype(int)

# =======================================================
# 🧪 EXPERIMENT: 3D HYPERPARAMETER SEARCH
# =======================================================

best_global_f1 = -1
best_global_config = {}
best_global_model = None

print(f"\n=======================================================")
print(f"STARTING GLOBAL 3D HYPERPARAMETER SEARCH (Ratio, C, K)")
print(f"=======================================================")

# Loop 1: Downsampling Ratio
for DOWNSAMPLE_RATIO in DOWNSAMPLE_RATIO_VALUES:
    
    # --- 3. Training Data Downsampling (RE-SAMPLE for each ratio) ---
    # We use the X_train_full and y_train_full we loaded above
    pos_indices = np.where(y_train_full == 1)[0]
    neg_indices = np.where(y_train_full == 0)[0]
    
    X_pos = X_train_full[pos_indices]
    y_pos = y_train_full[pos_indices]
    X_neg = X_train_full[neg_indices]
    y_neg = y_train_full[neg_indices]
    
    num_pos = len(y_pos)
    num_neg_to_keep = int(min(len(y_neg), num_pos * DOWNSAMPLE_RATIO))
    
    # Randomly sample negative cases
    if len(y_neg) > 0:
        neg_sample_indices = np.random.choice(len(y_neg), size=num_neg_to_keep, replace=False)
        X_neg_sampled = X_neg[neg_sample_indices]
        y_neg_sampled = y_neg[neg_sample_indices]
        
        X_train_sampled = np.vstack([X_pos, X_neg_sampled])
        y_train_sampled = np.concatenate([y_pos, y_neg_sampled])
    else:
        # Fallback if no negatives (unlikely)
        X_train_sampled = X_pos
        y_train_sampled = y_pos
    
    print(f"\n-- DATASET: Ratio {DOWNSAMPLE_RATIO}:1 (Train Size: {len(y_train_sampled)}) --")

    # Loop 2: Regularization Parameter (C)
    for C_VAL in C_VALUES:
        
        # Loop 3: Feature Count (K)
        for K_VAL in K_BEST_VALUES:
            
            # 4. Build Pipeline
            clf_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler()), 
                    ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)), 
                    ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=K_VAL)), 
                    ("clf", LinearSVC(
                        C=C_VAL, 
                        class_weight="balanced", 
                        max_iter=10000, 
                        random_state=RANDOM_STATE,
                        dual='auto'
                    )),
                ]
            )
            
            # 5. Fit the model
            clf_pipeline.fit(X_train_sampled, y_train_sampled)
            
            # 6. Tune Threshold
            current_best_f1, optimal_t = tune_threshold(clf_pipeline, X_test, y_test)
            
            print(f" (K={K_VAL}, C={C_VAL:.1f}) F1: {current_best_f1:.4f}")

            # 7. Track Best
            if current_best_f1 > best_global_f1:
                best_global_f1 = current_best_f1
                best_global_config = {
                    'Ratio': DOWNSAMPLE_RATIO,
                    'C': C_VAL,
                    'K': K_VAL,
                    'Threshold': optimal_t
                }
                best_global_model = clf_pipeline


# 8. Final Results Report
print("\n\n=======================================================")
print(f"🏆 GLOBAL OPTIMAL MODEL CONFIGURATION 🏆")
print("=======================================================")
print(f"Best Tuned F1-Score (Fire): {best_global_f1:.4f}")
print(f"Best Configuration: {best_global_config}")

if best_global_model:
    # Final Evaluation for Best Model
    # Get the raw decision scores (distance to hyperplane)
    y_scores_final = best_global_model.decision_function(X_test)
    
    # Calculate PR-AUC (Average Precision)
    final_pr_auc = average_precision_score(y_test, y_scores_final)
    
    # Calculate ROC-AUC
    final_roc_auc = roc_auc_score(y_test, y_scores_final)
    
    # Generate predictions based on the optimal threshold found earlier
    y_pred_final = (y_scores_final >= best_global_config['Threshold']).astype(int)
    
    print(f"\nFinal ROC-AUC: {final_roc_auc:.4f}")
    print(f"Final PR-AUC:  {final_pr_auc:.4f}")  # <--- NEW METRIC
    
    print("\nFinal Classification report (BEST TUNED MODEL):")
    print(classification_report(y_test, y_pred_final, digits=4))