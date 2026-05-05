# %% [markdown]
# # Evaluation and Analysis
# 
# This notebook applies 6 different performance metrics and 6 visualization techniques for the ML classifiers we have built for our project. Our analysis focuses on the overall model performance and age-group bias, specifically whether the models perform worse for younger patients (Under 40) compared to older patients.
# 
# **Target variable:** `target_michd` - whether a respondent has ever been diagnosed with myocardial infarction (MI) or coronary heart disease (CHD).
# 
# The following models are evaluated in this notebook:
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - XGBoost
# - Support Vector Machine (SVM)
# 
# ---
# 
# **Performance Metrics**
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1-Score
# 5. ROC-AUC
# 6. False Negative Rate
# 
# **Visualization Techniques**
# 1. Performance Metrics (Grouped Bar Chart)
# 2. ROC Curves
# 3. Precision-Recall Curves
# 4. Confusion Matrices
# 5. Recall by Age Group (Grouped Bar Chart)
# 6. False Negative Rate (FNR) by Age Group (Grouped Bar Chart)
# 
# ---

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score 
    )

# Ignore warnings
warnings.filterwarnings('ignore')

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

# Instead of the full model name, use abbreviations
label_map = {
    'Logistic Regression (Baseline)': 'LR (Baseline)',
    'Logistic Regression (FFS)': 'LR (FFS)',
    'Logistic Regression (RFS)': 'LR (RFS)',
    'Decision Tree (Baseline)': 'DT (Baseline)',
    'Decision Tree (FFS)': 'DT (FFS)',
    'Decision Tree (RFS)': 'DT (RFS)',
    'Random Forest (Baseline)': 'RF (Baseline)',
    'Random Forest (FFS)': 'RF (FFS)',
    'Random Forest (RFS)': 'RF (RFS)',
    'XGBoost (Baseline)': 'XGB (Baseline)',
    'XGBoost (FFS)': 'XGB (FFS)',
    'XGBoost (RFS)': 'XGB (RFS)',
    'Support Vector Machine (Baseline)': 'SVM (Baseline)',
    'Support Vector Machine (FFS)': 'SVM (FFS)',
    'Support Vector Machine (RFS)': 'SVM (RFS)',
}


# %% [markdown]
# ## 2. Load test set
# 
# The test sets and selector were saved from our modeling notebook to avoid duplilcate data preparation code.
# 
# We use index_col=0 to restore the original row index. This is required to align the age group labels with the right test rows.

# %%
REPO = Path.cwd().parent
(REPO / 'visualizations').mkdir(exist_ok=True)

# Load test data
X_test = pd.read_csv(REPO/'data'/'processed'/'X_test_full.csv', index_col=0)
X_test_reduced = pd.read_csv(REPO/'data'/'processed'/'X_test_reduced.csv', index_col=0)
y_test = pd.read_csv(REPO/'data'/'processed'/'y_test.csv', index_col=0).squeeze() # Convert to Series

# Load training data (for age group distribution analysis)
X_train_reduced = pd.read_csv(REPO/'data'/'processed'/'X_train_reduced.csv', index_col=0)
y_train = pd.read_csv(REPO/'data'/'processed'/'y_train.csv', index_col=0) .squeeze()

selector = pickle.load(open(REPO/'models'/'feature_selector.model', 'rb'))

print("X_test shape:", X_test.shape)
print("X_test_reduced shape:", X_test_reduced.shape)
print("y_test shape:", y_test.shape)
print("Class Distribution:")
print(y_test.value_counts(normalize=True).rename({0: 'No Heart Disease', 1: 'Heart Disease'}))

# %% [markdown]
# ### Abbreviations
# 
# - **Baseline**: Unweighted model trained on the full feature set (83 one-hot encoded columns, 25 features).
# - **FFS (Full Feature Set)**: Model trained on all 25 features with `class_weight='balanced'`.
# - **RFS (Reduced Feature Set)**: Model trained on reduced features selected via `SelectKBest` (mutual information) with `class_weight='balanced'`.
# 
# `class_weight='balanced'` re-weights training samples so minority class (heart disease)
# errors are penalized more heavily, addressing the ~9:1 class imbalance.

# %% [markdown]
# ## 3. Load saved models

# %%
# Define model file paths
MODELS_DIR = REPO/'models'

model_configs = [
    ('Logistic Regression (Baseline)', 'lr_baseline.model', X_test),
    ('Logistic Regression (FFS)', 'lr_full.model', X_test), 
    ('Logistic Regression (RFS)', 'lr_reduced.model', X_test_reduced),

    ('Decision Tree (Baseline)', 'dt_baseline.model', X_test),
    ('Decision Tree (FFS)', 'dt_full.model', X_test),
    ('Decision Tree (RFS)', 'dt_reduced.model', X_test_reduced),

    ('Random Forest (Baseline)', 'rf_baseline.model', X_test),
    ('Random Forest (FFS)', 'rf_full.model', X_test),
    ('Random Forest (RFS)', 'rf_reduced.model', X_test_reduced),

    ('XGBoost (Baseline)', 'xgb_baseline.model', X_test),
    ('XGBoost (FFS)', 'xgb_full.model', X_test),
    ('XGBoost (RFS)', 'xgb_reduced.model', X_test_reduced),

    ('Support Vector Machine (Baseline)', 'svm_baseline.model', X_test),
    ('Support Vector Machine (FFS)', 'svm_full.model', X_test),
    ('Support Vector Machine (RFS)', 'svm_reduced.model', X_test_reduced),
]

# Load models
models = []
for model_name, file_name, X_eval in model_configs:
    model = pickle.load(open(MODELS_DIR/file_name, 'rb'))
    models.append((model_name, model, X_eval))

print(f"Loaded {len(models)} models.")



# %% [markdown]
# ## 4. Make predictions using all models

# %%
predictions = []

for model_name, model, X_eval in models:
    y_predict = model.predict(X_eval) # Target class predictions (0: No heart disease, 1: Heart disease)
    y_prob = model.predict_proba(X_eval)[:, 1] # Only get probability of positive class (heart disease)
    predictions.append((model_name, y_predict, y_prob))

print(f"Generated predictions for {len(predictions)} models.")

# %% [markdown]
# ## 5. Apply performance metrics
# 1. **Accuracy**: proportion of correct predictions over all predictions.
# 2. **Precision**: proportion of all model's positive predictions that are actually positive. Of all cases that are predicted as heart disease, how many actually have it.
# 3. **Recall**: Of all actual heart disease cases, how many did the model catch?
# 4. **F1-Score**: Harmonic mean of Precision and Recall. A higher F1-scores indicates better model performance. (Useful for imbalanced classes)
# 5. **ROC-AUC**: evaluates how well a binary classification model works. (0.5=random guessing, 1.0=perfect) 
# 6. **False Negative Rate (FNR)**: proportion of positive heart disease cases the model missed.

# %%
metrics_results = []
for model_name, y_pred, y_prob in predictions:
    # Unpacks true negative (tn), false positive (fp), false negative (fn), and true positive (tp)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate performance metrics
    metrics_results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'False Negative Rate (FNR)': fn/(fn+tp)
    })

def color_rows(row):
    if 'Baseline' in row['Model']:
        return ['background-color: darkred']*len(row)
    return ['']*len(row)

metrics_df = pd.DataFrame(metrics_results).reset_index(drop=True)
metrics_df.style.apply(color_rows, axis=1).format(precision=4)

# %% [markdown]
# ## 6. Visualizations
# 1. Grouped Bar Chart
# 2. ROC Curves
# 3. Confusion Matrices
# 
# 
# 

# %% [markdown]
# ### 6.1 Grouped Bar Chart
# *We exclude False Negative Rate from the grouped bar chart since it's a different metric, where lower values are better*.

# %%
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

# Split metric results into separate DataFrames for each model type
baseline_df = metrics_df[metrics_df['Model'].str.contains('Baseline')]
ffs_df = metrics_df[metrics_df['Model'].str.contains('FFS')]
rfs_df = metrics_df[metrics_df['Model'].str.contains('RFS')]

# Plot grouped bar charts for each model type
for df, title, filename in zip(
    [baseline_df, ffs_df, rfs_df],
    ['Baseline (Unweighted)', 'FFS', 'RFS'],
    ['grouped_bar_chart_baseline.png', 'grouped_bar_chart_ffs.png', 'grouped_bar_chart_rfs.png']
):
    df = df.copy() 
    df['Model'] = df['Model'].map(label_map)    # abbreviate model names using label_map

    fig, axes = plt.subplots(figsize=(7, 3.5))
    melted = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', ax=axes)

    axes.set_title(f'Performance Metrics for {title} Models')
    axes.set_xlabel('')
    axes.set_ylabel('Score')
    axes.set_ylim(0, 1.00)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPO/'visualizations'/filename)
    plt.show()

# %% [markdown]
# ### 6.2 ROC Curves
# The ROC curve plots True Positive Rate (TPR) (y-axis) vs False Positive Rate (FPR) (x-axis) at different classification thresholds. The Area Under the Curve (AUC) measures overall model performance. An AUC score of 0.50 represents random guessing, and 1.0 is the best possible score, indicating perfect classification.
# 
# XGBoost with the full feature set showed the highest AUC score of 0.8289.
# 
# **True Positive Rate**: From all the actual positive cases, how many were correctly identified?
# 
# **False Positive Rate**: From all the actual negative cases, how many were incorrectly classified as positive?

# %%
fig, axes = plt.subplots()

for model_name, y_pred, y_prob in predictions:
    if 'RFS' in model_name or 'Baseline' in model_name: # Skip RFS and Baseline models for clearer visualization
        continue

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes.plot(fpr, tpr, label=f"{label_map[model_name]} (AUC = {auc:.4f})")

axes.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
axes.set_xlabel('False Positive Rate') 
axes.set_ylabel('True Positive Rate (Recall)')
axes.set_title('ROC Curves for Weighted FFS Models')
axes.legend(loc='lower right')
plt.tight_layout()
plt.savefig(REPO/'visualizations'/'roc_curves.png')
plt.show()


# %% [markdown]
# ### 6.3 Precision-Recall Curve
# 

# %%
fig, axes = plt.subplots()

for model_name, y_pred, y_prob in predictions:
    if 'RFS' in model_name or 'Baseline' in model_name: # skip RFS and Baseline models
        continue
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    axes.plot(recall, precision, label=f"{label_map[model_name]} (AP = {avg_precision:.3f})")

# Baseline reference line
baseline = y_test.mean() # Proportion of positive class in the test set
print(f'{baseline*100:.3f}% of test samples have heart disease (positive class).')
axes.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')

axes.set_title('Precision-Recall Curves')
axes.set_xlabel('Recall')
axes.set_ylabel('Precision')
axes.legend(loc='upper right')
plt.tight_layout()
plt.savefig(REPO/'visualizations'/'precision_recall_curves.png')
plt.show()

# %% [markdown]
# ### 6.4 Confusion Matrices
# Plot confusion matrices for unweighted baseline models and weighted FFS models.

# %%
cm_model_configs = [
    ('Logistic Regression (Baseline)', 'Reds', 'cm_logistic_baseline.png'),
    ('Logistic Regression (FFS)', 'Blues', 'cm_logistic_ffs.png'),
    ('Logistic Regression (RFS)', 'Greens', 'cm_logistic_rfs.png'),

    ('Decision Tree (Baseline)', 'Reds', 'cm_dt_baseline.png'),
    ('Decision Tree (FFS)', 'Blues', 'cm_dt_ffs.png'),
    ('Decision Tree (RFS)', 'Greens', 'cm_dt_rfs.png'),

    ('Random Forest (Baseline)', 'Reds', 'cm_rf_baseline.png'),
    ('Random Forest (FFS)', 'Blues', 'cm_rf_ffs.png'),
    ('Random Forest (RFS)', 'Greens', 'cm_rf_rfs.png'),

    ('XGBoost (Baseline)', 'Reds', 'cm_xgb_baseline.png'),
    ('XGBoost (FFS)', 'Blues', 'cm_xgb_ffs.png'),
    ('XGBoost (RFS)', 'Greens', 'cm_xgb_rfs.png'),

    ('Support Vector Machine (Baseline)', 'Reds', 'cm_svm_baseline.png'),
    ('Support Vector Machine (FFS)', 'Blues', 'cm_svm_ffs.png'),
    ('Support Vector Machine (RFS)', 'Greens', 'cm_svm_rfs.png'),

]

pred_dict = {model_name: y_pred for model_name, y_pred, y_prob in predictions}

for model_name, cmap, filename in cm_model_configs:
    fig, axes = plt.subplots()
    cm = confusion_matrix(y_test, pred_dict[model_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True, ax=axes,
                xticklabels=['No Heart Disease', 'Heart Disease'], 
                yticklabels=['No Heart Disease', 'Heart Disease'])
    
    axes.set_title(model_name)   
    axes.set_xlabel('Predicted Label', fontweight='bold')
    axes.set_ylabel('True Label', fontweight='bold')

    plt.tight_layout()
    plt.savefig(REPO/'visualizations'/filename)
    plt.show()

# %% [markdown]
# ## Age Group Analysis

# %%
age_cols = [col for col in X_test.columns if 'age_group=' in col.lower()]
print(age_cols)

# %%

# Get all age one hot columns from the FFS
age_cols = [col for col in X_test.columns if col.startswith('age_group=')]

# For each test row, find which age bracket column is 1
age_raw = X_test[age_cols].idxmax(axis=1).str.replace('age_group=', '', regex=False)

# Create 3 age groups: Under 40, 40-55, Over 55
young_bracket  = ['18-24', '25-29', '30-34', '35-39']
middle_bracket = ['40-44', '45-49', '50-54']
# over 55: 55-59, 60-64, 65-69, 70-74, 75-79, 80+

def map_age_group(age):
    if age in young_bracket:  return 'Under 40'
    if age in middle_bracket: return '40-55'
    return 'Over 55'

age_labels = age_raw.map(map_age_group)

# Verify: show count and heart disease rate per age group
age_summary = pd.DataFrame({'Age Group': age_labels, 'target': y_test})
age_order = ['Under 40', '40-55', 'Over 55']
summary = age_summary.groupby('Age Group')['target'].agg(['count', 'mean']).rename(
    columns={'count': 'Test Samples', 'mean': 'Heart Disease Rate'}).reindex(age_order).reset_index()

summary['Heart Disease Rate'] = (summary['Heart Disease Rate'] * 100).round(2).astype(str) + '%'
summary


# %%
age_cols = [col for col in X_test.columns if col.startswith('age_group=')]

young_bracket  = ['18-24', '25-29', '30-34', '35-39']
middle_bracket = ['40-44', '45-49', '50-54']
age_order = ['Under 40', '40-55', 'Over 55']

X_all = pd.concat([X_train_reduced[age_cols], X_test[age_cols]])
y_all = pd.concat([y_train, y_test])

def map_age_group(age):
    if age in young_bracket:  return 'Under 40'
    if age in middle_bracket: return '40-55'
    return 'Over 55'

def hd_rate_by_age(X, y):
    age_raw = X[age_cols].idxmax(axis=1).str.replace('age_group=', '', regex=False)
    df = pd.DataFrame({'age_group': age_raw.map(map_age_group), 'target': y})
    return df.groupby('age_group')['target'].mean()

result = pd.DataFrame({
    'Age Group': age_order,
    'Heart Disease Rate (Full)':  hd_rate_by_age(X_all, y_all).reindex(age_order).values,
    'Heart Disease Rate (Train)': hd_rate_by_age(X_train_reduced, y_train).reindex(age_order).values,
    'Heart Disease Rate (Test)':  hd_rate_by_age(X_test, y_test).reindex(age_order).values
})

rate_cols = ['Heart Disease Rate (Full)', 'Heart Disease Rate (Train)', 'Heart Disease Rate (Test)']
result.style.format({col: '{:.2%}' for col in rate_cols})

# %%
results = []

for model_name, y_pred, y_prob in predictions:
    pred = pd.Series(y_pred, index=X_test.index)
    for age_group in ['Under 40', '40-55', 'Over 55']:
        index = age_labels == age_group
        y_sub = y_test[index]
        y_pred_sub = pred[index]
        recall = recall_score(y_sub, y_pred_sub)
        fnr = 1 - recall

        results.append({
            'Model': label_map[model_name],
            'Age Group': age_group,
            'Recall': recall,
            'False Negative Rate (FNR)': fnr
        })

def color_rows(row):
    if 'Baseline' in row.name:
        return ['background-color: darkred']*len(row)
    return ['']*len(row)

results_df = pd.DataFrame(results)
# Instead of the full model name, use abbreviations
results_df['Model'] = results_df['Model']\
    .str.replace('Logistic Regression', 'LR')\
    .str.replace('Decision Tree', 'DT')\
    .str.replace('Random Forest', 'RF')\
    .str.replace('XGBoost', 'XGB')
pivot = results_df.pivot(index='Model', columns='Age Group', values='Recall')
pivot = pivot[['Under 40', '40-55', 'Over 55']]
pivot.style.apply(color_rows, axis=1).format(precision=4).set_caption("Recall by Age Group")


# %% [markdown]
# ### 6.5 Recall by Age Group 

# %%

baseline_age_df = results_df[results_df['Model'].str.contains('Baseline')]
ffs_age_df = results_df[results_df['Model'].str.contains('FFS')]
rfs_age_df = results_df[results_df['Model'].str.contains('RFS')]


# Plot grouped bar charts for each model type
for df, title, filename in zip(
    [baseline_age_df, ffs_age_df, rfs_age_df],
    ['Baseline (Unweighted)', 'FFS', 'RFS'],
    ['recall_by_age_group_baseline.png', 'recall_by_age_group_ffs.png', 'recall_by_age_group_rfs.png']
):
    
    fig, axes = plt.subplots(figsize=(7, 3.5))
    # melted = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    sns.barplot(data=df, x='Age Group', y='Recall', hue='Model', ax=axes)
    
    for age_group in axes.containers:
        axes.bar_label(age_group, fmt='%.2f')

    axes.set_title(f'Recall by Age Group for {title} Models')
    axes.set_xlabel('Age Group')
    axes.set_ylabel('Recall Score')
    axes.set_ylim(0, 1.00)
    axes.legend(loc='upper center', ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPO/'visualizations'/filename)
    plt.show()

# %% [markdown]
# ### 6.6 False Negative Rate (FNR) by Age Group

# %%
pivot = results_df.pivot(index='Model', columns='Age Group', values='False Negative Rate (FNR)')
pivot = pivot[['Under 40', '40-55', 'Over 55']]
pivot.style.apply(color_rows, axis=1).format(precision=4).set_caption("False Negative Rate by Age Group")

# %%

baseline_age_df = results_df[results_df['Model'].str.contains('Baseline')]
ffs_age_df = results_df[results_df['Model'].str.contains('FFS')]
rfs_age_df = results_df[results_df['Model'].str.contains('RFS')]


# Plot grouped bar charts for each model type
for df, title, filename in zip(
    [baseline_age_df, ffs_age_df, rfs_age_df],
    ['Baseline (Unweighted)', 'FFS', 'RFS'],
    ['fnr_by_age_group_baseline.png', 'fnr_by_age_group_ffs.png', 'fnr_by_age_group_rfs.png']
):
    fig, axes = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=df, x='Age Group', y='False Negative Rate (FNR)', hue='Model', ax=axes)

    for age_group in axes.containers:
        axes.bar_label(age_group, fmt='%.2f')

    axes.set_title(f'False Negative Rate by Age Group for {title} Models')
    axes.set_xlabel('Age Group')
    axes.set_ylabel('False Negative Rate')
    axes.set_ylim(0, 1.10)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPO/'visualizations'/filename)
    plt.show()

# %% [markdown]
# ## Analysis
# 
# ### **Overall Model Comparison**
# 
# All five models (Logistic Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Machine) were evaluated on the test set which consists of 90,489 respondents using the following six performance metrics: accuracy, precision, recall, F1-score, ROC-AUC, and false negative rate (FNR). The baseline (unweighted) models score approximately 90% accuracy across all four models. With class weighting, accuracy scores range from 70% to 72% for both the full feature set (FFS) and reduced feature set (RFS) models. There is a tradeoff between accuracy and recall after applying class weighting. However, accuracy alone is not a reliable metric to evaluate our models due to the 9:1 imbalanced class ratio, where roughly 90% of all cases have 'No Heart Disease', and the other 10% have 'Heart Disease'. The baseline models are predicting the majority class. Due to the imbalanced dataset, other performance metrics provide more clarity into our model performance.
# 
# Precision is less than 23% across all weighted models, revealing that 77% of heart disease predictions are false positives. This is a tradeoff when optimizing recall for imbalanced datasets. As precision declines, recall increases at the risk of having more false alarms.
# 
# XGBoost (on the full feature set) shows the best performance overall, resulting in the best ROC-AUC (0.8289), high Recall (0.7913), and lowest False Negative Rate (0.2087) among all models. XGBoost correctly identified 79% of true heart disease cases in the test set. Support Vector Machine (SVM) is a strong runner-up with an ROC-AUC at 0.8221, highest recall (0.8241), and lowest FNR (0.1759).
# 
# 
# ### **Age Group Bias Analysis**
# 
# Three different age groups were created (Under 40, 40-55, Over 55) to evaluate model performance based on age. Our results confirm that heart disease prediction models perform significantly worse for younger patients.
# 
# For each age group, the rate of positive heart disease cases are 1.08% for individuals under 40, 4.16% for individuals from 40-55, and 14.72% for individuals over 55. Our models were trained on more positive heart disease cases where the patient was over 55. During training, all models were exposed to fewer examples of positive heart disease cases where age was under 40. Due to the class imbalance, the models struggle learning the risk patterns associated with heart disease for younger individuals.
# 
# The **Recall by Age Group** graph visually depicts the significant gap in recall between each age group. For the 'Over 55' age group, all models achieved a recall of over 80%. On the other hand, the 'Under 40' age group showcased a significant decline in recall, ranging from 7.26% (Logistic Regression) to 15.81% (Decision Tree). This means these models miss between 84.19% to 92.74% of all positive heart disease cases in younger individuals under 40. 
# 
# The findings shown on the **False Negative Rate (FNR) by Age Group** are clinically concerning. For all baseline models, the false negative rate is over 92% for all models, and approximately 100% for the 'Under 40' age group. The model classifies individuals as 'No Heart Disease' and misses actual Heart Disease cases. After applying class weighting to our models, the FNR declined drastically for the 'Over 55' age group, dropping from 92% to 16%. For the '40-55' age group, FNR dropped to 52-56%, showing partial improvement but still missing over half of positive heart disease cases. However, the 'Under 40' age group has a false negative rate (FNR) that ranges from 84-93% after applying class weighting. Due to the low prevalence of positive cases for the 'Under 40' age group (1.08%), the models do not have enough examples to learn from. This pattern is visible across all models, indicating an age imbalance in the data. The machine learning models do not perform equally across all age groups, and younger individuals are more likely to be classified as low risk.
# 
# 
# ### **Key Findings**
# - Weighted XGBoost on the FFS achieves the best overall performance (Recall: 79.1%, FNR: 20.9%)
# - Class weighting significantly improves recall for all models (weighted vs. baseline)
# - Age bias remains even after balanced class weighting ('Under 40' FNR remains at 84-93%)
# - Simple class reweighting is not enough for handling the class imbalance and age bias. (Additional data on younger patients with heart disease may be needed to address this limitation.)

# %% [markdown]
# ***References***
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score
# - https://www.ibm.com/docs/en/ws-and-kc?topic=metrics-false-negative-rate-difference
# - https://www.ibm.com/docs/en/ws-and-kc?topic=metrics-area-under-roc


