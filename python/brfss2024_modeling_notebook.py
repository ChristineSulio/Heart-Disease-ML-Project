# %% [markdown]
# # BRFSS 2024: Heart Disease ML Modeling
# 
# This notebook implements five different machine learning techniques for our heart disease classification project.
# 
# The following ML techniques were implemented:
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - XGBoost
# - Support Vector Machine (SVM)
# 
# There are three models produced for each ML technique: an unweighted baseline, a class-weighted full feature set (FFS), and a class-weighted reduced feature set (RFS).
# 
# **Target variable:** `target_michd` - whether a respondent has ever been diagnosed with myocardial infarction (MI) or coronary heart disease (CHD).

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

# Set global plot styles
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,          
    'axes.titlesize': 9,     
    'axes.labelsize': 9,
    'xtick.labelsize': 8,    
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 8,
    'legend.loc': 'upper center',
    'figure.dpi': 150,
    'savefig.dpi': 300,      
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.figsize': (3.5, 3.0) 
})

# %% [markdown]
# ## 2. Load preprocessed data

# %%
REPO = Path.cwd().parent
ML_CSV = REPO / 'data' / 'processed' / 'brfss2024_heart_bias_preprocessed_ml.csv'
MODELS_DIR = REPO / 'models'
MODELS_DIR.mkdir(exist_ok=True)
(REPO / 'data' / 'processed').mkdir(parents=True, exist_ok=True)

print(f"Model files will be saved to: {MODELS_DIR}")
df = pd.read_csv(ML_CSV)
print(f"Loaded dataset shape: {df.shape}")

target_dist = ( df['target_michd']     
    .value_counts(normalize=True)
    .rename({0:'No Heart Disease', 1:'Heart Disease'})
    .reset_index()     
    )

target_dist.columns = ['Heart Disease Status', 'Proportion']
target_dist.style.set_caption("Target Variable Distribution")
target_dist

# %% [markdown]
# ## 3. Separate features and target
# We exclude our target variable **`target_michd`** and the BRFSS survey sampling weight **`sample_weight`** from X.
# 
# *The `sample_weight` is a survey sampling weight which represents how many Americans each respondent represents in the overall US population.*

# %%
X = df.drop(columns=['target_michd', 'sample_weight'])
y = df['target_michd'] # Output target variable (1 = has heart disease, 0 = no heart disease) 

print("X shape:", X.shape) 
print("y shape:", y.shape) 

# %% [markdown]
# ## 4. Split data into train/test sets

# %%
# Use stratified sampling to preserve the class imbalance ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Train size: {X_train.shape}')
print(f'Test size: {X_test.shape}')

# Verify stratification worked - should have similar class proportions for target
print(f"y_train positive class ratio: {y_train.mean():.4f}")
print(f"y_test positive class ratio: {y_test.mean():.4f}")

# %% [markdown]
# ## 5. Feature Selection
# - Mutual Information Score by Feature Rank plot - to visualize best k value
# - Cross validation - to determine best k based on average precision
# 
# *(Note: The cell below takes about 2 minutes to complete.)*

# %%
# Score all one-hot columns using mutual information score
mutual_info_scorer = partial(mutual_info_classif, random_state=42) 
selector = SelectKBest(mutual_info_scorer, k='all')
selector.fit(X_train, y_train)

scores_df = pd.DataFrame({
    'feature': X_train.columns,
    'mutual_info_score': selector.scores_
}).sort_values(by='mutual_info_score', ascending=False).reset_index(drop=True).round(4)

# Aggregate mutual information scores by original feature (sum across equivalent one-hot columns)
scores_df['original_feature'] = scores_df['feature'].apply(lambda x: x.split('=')[0] if '=' in x else x) # Get original feature name before one-hot encoding

feature_scores_df = (scores_df
    .groupby('original_feature')['mutual_info_score']
    .sum()
    .reset_index()
    .sort_values(by='mutual_info_score', ascending=False)
)

feature_scores_df.columns = ['feature', 'mutual_info_score']
ranked_original_features = feature_scores_df['feature'].tolist()

# Plot graph sorted by aggregated mutual information score
plt.figure()
plt.plot(range(1, len(feature_scores_df) + 1), feature_scores_df['mutual_info_score'], marker='o')

plt.xlabel('Feature Rank (1 = most informative)')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Score by Feature Rank')
plt.grid(True)
plt.xticks(range(0, len(feature_scores_df) + 1, 5))
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cross-validation

# %%
# Get ranked features and test different k values for cross-validation score
best_k = 0
best_score = 0
cv_results = []

print(f"Testing cross-validation scores (using average precision) for {len(ranked_original_features)} original features...")
print(f"k ->\tCV avg precision score")

for k in range (1, len(ranked_original_features) + 1):
    top_k = ranked_original_features[:k]
    selected_cols = [col for col in X_train.columns if col.split('=')[0] in top_k or col in top_k]
    X_temp = X_train[selected_cols]
    score = cross_val_score(LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
                            X_temp, y_train, cv=3, scoring='average_precision', n_jobs=-1).mean()    
    
    print(f"{k} ->\t{score:.4f}")
    cv_results.append({'k': k, 'cv_average_precision': score})
    
    if score > best_score + 0.01:  # Only update if there's a meaningful improvement
        best_score = score
        best_k = k

cv_results_df = pd.DataFrame(cv_results)
print(f"\nBest k: {best_k},\tCV score: {best_score:.4f}")
cv_results_df.style.hide(axis='index').format({'cv_average_precision': '{:.4f}'}).set_caption("Cross-Validation Results")

# %% [markdown]
# ### Create reduced feature set (RFS)

# %%
# Select top k original features based on aggregated mutual information scores
top_features = ranked_original_features[:best_k]
selected_features = [col for col in X_train.columns if col.split('=')[0] in top_features or col in top_features]
X_train_reduced = X_train[selected_features]
X_test_reduced  = X_test[selected_features]

print(f"X_train_reduced: {X_train_reduced.shape}")
print(f"X_test_reduced: {X_test_reduced.shape}")

print(f"Selected {best_k} features")
feature_scores_df.head(best_k).reset_index(drop=True)


# %% [markdown]
# ### One-Hot Encoding vs Original Features
# Mutual Information Scores are computed for each one-hot encoded column, then summed by the original feature. Features are ranked by their total mutual information score and cross-validation selects the best number of original features to keep. All one-hot columns belonging to the selected original features are included in the reduced feature set (RFS).
# 
# The *Feature Selection Summary* table below shows which features were kept and removed.
# 

# %%
# Get original feature names
all_features = set(col.split('=')[0] for col in X_train.columns)
kept_features = set(col.split('=')[0] for col in selected_features)
removed_features = all_features - kept_features
print(f"Reduced feature set: {best_k} original features -> {len(selected_features)} one-hot encoded columns")

# Display summary of which original features were kept vs removed
pd.concat([
    pd.DataFrame({'feature': sorted(kept_features), 'status': 'kept'}),
    pd.DataFrame({'feature': sorted(removed_features), 'status': 'removed'})
]).sort_values('feature').reset_index(drop=True).style.set_caption("Feature Selection Summary")



# %% [markdown]
# ## 6. ML model training
# Implements the following ML techniques: Logistic Regression, Decision Tree, Random Forest, and XGBoost
# 1. Train baseline model
# 2. Train weighted model on full feature set (FFS)
# 3. Train weighted model on reduced feature set (RFS)
# 

# %% [markdown]
# ### 6.1 Logistic Regression

# %%
# Unweighted baseline model - using full feature set
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train, y_train)
pickle.dump(lr_baseline,  open(MODELS_DIR/'lr_baseline.model',  'wb'))

# Weighted model with FFS
lr_full = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_full.fit(X_train, y_train)
pickle.dump(lr_full, open(MODELS_DIR/'lr_full.model', 'wb'))

# Weighted model with RFS
lr_reduced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_reduced.fit(X_train_reduced, y_train)
pickle.dump(lr_reduced, open(MODELS_DIR/'lr_reduced.model', 'wb'))

# %% [markdown]
# ### 6.2 Decision Tree

# %%
# Unweighted baseline model - using full feature set
dt_baseline = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_baseline.fit(X_train, y_train)
pickle.dump(dt_baseline,  open(MODELS_DIR/'dt_baseline.model',  'wb'))

# Weighted model with FFS
dt_full = DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42)
dt_full.fit(X_train, y_train)
pickle.dump(dt_full, open(MODELS_DIR/'dt_full.model', 'wb'))

# Weighted model with RFS
dt_reduced = DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42)
dt_reduced.fit(X_train_reduced, y_train)
pickle.dump(dt_reduced, open(MODELS_DIR/'dt_reduced.model', 'wb'))

# %% [markdown]
# ### 6.3 Random Forest

# %%
# Unweighted baseline model - using full feature set
rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
pickle.dump(rf_baseline,  open(MODELS_DIR/'rf_baseline.model',  'wb'))

# Weighted model with FFS
rf_full = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_full.fit(X_train, y_train)
pickle.dump(rf_full, open(MODELS_DIR/'rf_full.model', 'wb'))

# Weighted model with RFS
rf_reduced = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reduced.fit(X_train_reduced, y_train)
pickle.dump(rf_reduced, open(MODELS_DIR/'rf_reduced.model', 'wb'))

# %% [markdown]
# ### 6.4 XGBoost

# %%
# Unweighted baseline model - using full feature set
xgb_baseline = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_baseline.fit(X_train, y_train)
pickle.dump(xgb_baseline, open(MODELS_DIR/'xgb_baseline.model', 'wb'))

# Calculate pos_weight for XGBoost to handle class imbalance
pos_weight = (y_train == 0).sum()/(y_train == 1).sum()
print(f"Calculated pos_weight for XGBoost: {pos_weight:.4f}")

# Weighted model with FFS
xgb_full = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_full.fit(X_train, y_train)
pickle.dump(xgb_full, open(MODELS_DIR/'xgb_full.model', 'wb'))

# Weighted model with RFS
xgb_reduced = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_reduced.fit(X_train_reduced, y_train)
pickle.dump(xgb_reduced, open(MODELS_DIR/'xgb_reduced.model', 'wb'))

# %% [markdown]
# ### 6.5 Support Vector Machine (SVM)
# SVM Modeling via SGD.

# %%
# Unweighted baseline model - using full feature set
svm_baseline = SGDClassifier(loss='modified_huber', random_state=42, n_jobs=-1)
svm_baseline.fit(X_train, y_train)
pickle.dump(svm_baseline, open(MODELS_DIR/'svm_baseline.model', 'wb'))

# Weighted model with FFS
svm_full = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42, n_jobs=-1)
svm_full.fit(X_train, y_train)
pickle.dump(svm_full, open(MODELS_DIR/'svm_full.model', 'wb'))

# Weighted model with RFS
svm_reduced = SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42, n_jobs=-1)
svm_reduced.fit(X_train_reduced, y_train)
pickle.dump(svm_reduced, open(MODELS_DIR/'svm_reduced.model', 'wb'))

# %% [markdown]
# ## 7. Summary

# %%
# Create summary table of accuracy scores for all models
summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM'],
    'Baseline (unweighted)': [
        round(accuracy_score(y_test, lr_baseline.predict(X_test)), 4),
        round(accuracy_score(y_test, dt_baseline.predict(X_test)), 4),
        round(accuracy_score(y_test, rf_baseline.predict(X_test)), 4),
        round(accuracy_score(y_test, xgb_baseline.predict(X_test)), 4),
        round(accuracy_score(y_test, svm_baseline.predict(X_test)), 4),
    ],
    'FFS (weighted)': [
        round(accuracy_score(y_test, lr_full.predict(X_test)), 4),
        round(accuracy_score(y_test, dt_full.predict(X_test)), 4),
        round(accuracy_score(y_test, rf_full.predict(X_test)), 4),
        round(accuracy_score(y_test, xgb_full.predict(X_test)), 4),
        round(accuracy_score(y_test, svm_full.predict(X_test)), 4),
    ],
    'RFS (weighted)': [
        round(accuracy_score(y_test, lr_reduced.predict(X_test_reduced)), 4),
        round(accuracy_score(y_test, dt_reduced.predict(X_test_reduced)), 4),
        round(accuracy_score(y_test, rf_reduced.predict(X_test_reduced)), 4),
        round(accuracy_score(y_test, xgb_reduced.predict(X_test_reduced)), 4),
        round(accuracy_score(y_test, svm_reduced.predict(X_test_reduced)), 4),
    ],
})

summary.style.set_caption("Model Accuracy Summary (Baseline, FFS, and RFS)")


# %% [markdown]
# ## 8. Save test sets for evaluation

# %%
# Save test sets for evaluation (full feature set and reduced feature set)
X_test.to_csv(REPO / 'data' / 'processed' / 'X_test_full.csv', index=True)
X_test_reduced.to_csv(REPO / 'data' / 'processed' / 'X_test_reduced.csv', index=True)

# Save test labels
y_test.to_csv(REPO / 'data' / 'processed' / 'y_test.csv', index=True)

# Save training set (for class reweighting analysis in evaluation)
X_train_reduced.to_csv(REPO / 'data' / 'processed' / 'X_train_reduced.csv', index=True)
y_train.to_csv(REPO / 'data' / 'processed' / 'y_train.csv', index=True)

print("Saved:")
print("X_test:", X_test.shape)
print("X_test_reduced:", X_test_reduced.shape)
print("y_test:", y_test.shape)
print("X_train_reduced:", X_train_reduced.shape)
print("y_train:", y_train.shape)


