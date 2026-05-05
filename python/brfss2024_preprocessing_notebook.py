# %% [markdown]
# 
# # 2024 BRFSS preprocessing for Heart Disease Bias Project
# 
# This notebook preprocesses the official CDC BRFSS 2024 annual survey data for our heart disease classification/bias project.
# 
# **Target variable:** `_MICHD` - CDC calculated indicator for whether a respondent has ever reported myocardial infarction (MI) or coronary heart disease (CHD).
# 
# ## What this notebook includes
# - Discover and visualize the data to gain insights
# - Look for correlations
# - Experiment with attribute combinations
# - Prepare the data for machine learning algorithms
#   - Data cleaning
#   - Handling text and categorical attributes
#   - Handling missing values
#   - Handling outliers
#   - Feature scaling
#   - Transform feature encoding
#   - Dimensionality reduction (PCA evaluation)
# 

# %% [markdown]
# ## 1. Setup

# %%
import json
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style='whitegrid')
BASE = Path(__file__).parent if '__file__' in dir() else Path.cwd()
ROOT = BASE.parent
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)
XPT_PATH = ROOT / 'data' / 'raw' / 'LLCP2024.XPT'

SELECTED_COLS = [
    '_MICHD','_SEX','_AGEG5YR','_RACEGR3','_INCOMG1','_EDUCAG','MARITAL','EMPLOY1',
    'GENHLTH','PHYSHLTH','MENTHLTH','MEDCOST1','CHECKUP1','EXERANY2','DIABETE4',
    'ECIGNOW3','DRNK3GE5','MAXDRNKS','DRNKANY6','_DRNKWK3','_BMI5','_SMOKER3',
    '_TOTINDA','_LLCPWT'
]

AGE_MAP = {1:'18-24',2:'25-29',3:'30-34',4:'35-39',5:'40-44',6:'45-49',7:'50-54',8:'55-59',9:'60-64',10:'65-69',11:'70-74',12:'75-79',13:'80+'}
RACE_MAP = {1:'White_non_Hispanic',2:'Black_non_Hispanic',3:'Other_non_Hispanic',4:'Multiracial_non_Hispanic',5:'Hispanic'}
INCOME_MAP = {1:'lt_15k',2:'15k_25k',3:'25k_35k',4:'35k_50k',5:'50k_100k',6:'100k_200k',7:'ge_200k'}
EDU_MAP = {1:'No_HS_Diploma',2:'HS_Graduate',3:'Some_College',4:'College_Graduate'}
MARITAL_MAP = {1:'Married',2:'Divorced',3:'Widowed',4:'Separated',5:'Never_Married',6:'Unmarried_Couple'}
EMPLOY_MAP = {1:'Employed_for_wages',2:'Self_employed',3:'Out_of_work_gt_1yr',4:'Out_of_work_lt_1yr',5:'Homemaker',6:'Student',7:'Retired',8:'Unable_to_work'}
GENHLTH_MAP = {1:'Excellent',2:'Very_Good',3:'Good',4:'Fair',5:'Poor'}
YES_NO_MAP = {1:'Yes',2:'No'}
CHECKUP_MAP = {1:'Within_past_year',2:'Within_past_2_years',3:'Within_past_5_years',4:'Five_or_more_years_ago',8:'Never'}
DIABETES_MAP = {1:'Yes',2:'Yes_during_pregnancy',3:'No',4:'Prediabetes'}
ECIG_MAP = {1:'Every_day',2:'Some_days',3:'Not_at_all',4:'Never_used'}
SMOKER_MAP = {1:'Current_every_day',2:'Current_some_days',3:'Former_smoker',4:'Never_smoked'}
ACTIVE_MAP = {1:'Active',2:'Inactive'}

CATEGORICAL_FEATURES = [
    'sex', 'age_group', 'race_ethnicity', 'income_group', 'education_group', 'marital_status',
    'employment_status', 'general_health', 'couldnt_afford_doctor', 'last_checkup',
    'exercise_30d', 'diabetes_status', 'ecig_status', 'smoker_status', 'physically_active'
]
NUMERIC_FEATURES = [
    'bmi', 'phys_bad_days', 'ment_bad_days', 'drinks_per_week', 'binge_days',
    'max_drinks_single_occasion', 'poor_health_burden', 'smoke_exercise_risk',
    'age_bmi_interaction', 'diabetes_bmi_interaction'
]

def clean_days(s):
    s = s.replace({88:0, 77:np.nan, 99:np.nan})
    return s.where((s >= 0) & (s <= 30), np.nan).astype('float32')

def clip_iqr_values(s):
    valid = s.dropna()
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return s.clip(lower, upper).astype('float32'), {'lower': lower, 'upper': upper}


# %%
# Set global plot styles
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# %% [markdown]
# ## 2. Load the raw BRFSS 2024 data

# %%

raw_df, _ = pyreadstat.read_xport(str(XPT_PATH), usecols=SELECTED_COLS, encoding='LATIN1')
print('Selected raw shape:', raw_df.shape)
raw_df.head()


# %% [markdown]
# ## 3. Data cleaning, recoding, and feature engineering

# %%

df = raw_df[raw_df['_MICHD'].isin([1, 2])].copy()
df['target_michd'] = df['_MICHD'].map({1:1, 2:0}).astype('int8')
df['sex'] = df['_SEX'].map({1:'Male',2:'Female'})
df['age_group'] = df['_AGEG5YR'].map(AGE_MAP)
df['age_group_ord'] = df['_AGEG5YR'].where(df['_AGEG5YR'].between(1,13), np.nan).astype('float32')
df['race_ethnicity'] = df['_RACEGR3'].map(RACE_MAP)
df['income_group'] = df['_INCOMG1'].map(INCOME_MAP)
df['education_group'] = df['_EDUCAG'].map(EDU_MAP)
df['marital_status'] = df['MARITAL'].map(MARITAL_MAP)
df['employment_status'] = df['EMPLOY1'].map(EMPLOY_MAP)
df['general_health'] = df['GENHLTH'].map(GENHLTH_MAP)
df['phys_bad_days'] = clean_days(df['PHYSHLTH'])
df['ment_bad_days'] = clean_days(df['MENTHLTH'])
df['couldnt_afford_doctor'] = df['MEDCOST1'].map(YES_NO_MAP)
df['last_checkup'] = df['CHECKUP1'].map(CHECKUP_MAP)
df['exercise_30d'] = df['EXERANY2'].map(YES_NO_MAP)

df['diabetes_status'] = df['DIABETE4'].map(DIABETES_MAP)
# Drop rows where Males have Yes_during_pregnancy (violates BRFSS survey logic)
# Per DIABETE4 codebook: pregnancy follow-up only asked to female respondents
invalid_diabetes = (df['sex'] == 'Male') & (df['diabetes_status'] == 'Yes_during_pregnancy')
df = df[~invalid_diabetes].copy()

df['ecig_status'] = df['ECIGNOW3'].map(ECIG_MAP)
df['smoker_status'] = df['_SMOKER3'].map(SMOKER_MAP)
df['physically_active'] = df['_TOTINDA'].map(ACTIVE_MAP)
df['bmi'] = (df['_BMI5'] / 100.0).where(lambda s: (s >= 10) & (s <= 100), np.nan).astype('float32')
df['drinks_per_week'] = (df['_DRNKWK3'].where(~df['_DRNKWK3'].isin([77700, 99900]), np.nan) / 100.0).astype('float32')
df['drinks_per_week'] = df['drinks_per_week'].where((df['drinks_per_week'] >= 0) & (df['drinks_per_week'] <= 500), np.nan)
df['binge_days'] = clean_days(df['DRNK3GE5'])
df['max_drinks_single_occasion'] = df['MAXDRNKS'].replace({77:np.nan, 99:np.nan}).where(lambda s: (s >= 0) & (s <= 76), np.nan).astype('float32')
df.loc[df['DRNKANY6'] == 2, 'max_drinks_single_occasion'] = 0
df['sample_weight'] = df['_LLCPWT'].astype('float32')
df['poor_health_burden'] = (df['phys_bad_days'].fillna(0) + df['ment_bad_days'].fillna(0)).astype('float32')
current_smoker = df['smoker_status'].isin(['Current_every_day', 'Current_some_days']).astype('int8')
inactive = df['physically_active'].eq('Inactive').astype('int8')
diabetes_yes = df['diabetes_status'].eq('Yes').astype('int8')
df['smoke_exercise_risk'] = (current_smoker * inactive).astype('float32')
df['age_bmi_interaction'] = (df['age_group_ord'] * df['bmi']).astype('float32')
df['diabetes_bmi_interaction'] = (diabetes_yes * df['bmi']).astype('float32')
clip_bounds = {}
for col in ['bmi','drinks_per_week','max_drinks_single_occasion','age_bmi_interaction']:
    df[col], clip_bounds[col] = clip_iqr_values(df[col])

keep_cols = ['target_michd', 'sample_weight'] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
clean_df = df[keep_cols].copy()
for col in CATEGORICAL_FEATURES:
    clean_df[col] = clean_df[col].astype('category')

print(clean_df.shape)
clean_df.head()


# %% [markdown]
# ## 4. Missing values review

# %%

missing_summary = clean_df.isna().mean().sort_values(ascending=False)
missing_summary.head(15)


# %% [markdown]
# ## 5. Discover and visualize the data

# %%

eda_df = clean_df.sample(n=min(50000, len(clean_df)), random_state=42)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
eda_df['target_michd'].value_counts(normalize=True).sort_index().rename({0:'No Heart Disease',1:'Heart Disease'}).plot(kind='bar', ax=axes[0,0], color=['#4C78A8','#E45756'])
axes[0,0].set_title('Target Class Distribution')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='x', rotation=0)
axes[0,0].set_ylabel('Proportion')

sex_prev = eda_df.groupby('sex', observed=False)['target_michd'].mean()
axes[0,1].bar(sex_prev.index.astype(str), sex_prev.values, color=['#4C78A8','#F58518'])
axes[0,1].set_title('Heart Disease Prevalence by Sex')
axes[0,1].set_ylabel('Prevalence')

race_prev = eda_df.groupby('race_ethnicity', observed=False)['target_michd'].mean().sort_values(ascending=False)
axes[1,0].bar(race_prev.index.astype(str), race_prev.values, color='#B279A2')
axes[1,0].set_title('Heart Disease Prevalence by Race/Ethnicity')
axes[1,0].tick_params(axis='x', rotation=20)
axes[1,0].set_ylabel('Prevalence')

sns.histplot(eda_df['bmi'].dropna(), bins=40, kde=True, ax=axes[1,1], color='#72B7B2')
axes[1,1].set_title('BMI Distribution')
axes[1,1].set_xlabel('BMI')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Correlations

# %%

eda_corr = pd.get_dummies(eda_df.drop(columns=['sample_weight']), dummy_na=False, dtype='float32')
target_corr = eda_corr.corr(numeric_only=True)['target_michd'].drop('target_michd').abs().sort_values(ascending=False).head(15)
target_corr


# %% [markdown]
# ## 7. Experimenting with attribute combinations

# %%

combo = eda_df.groupby(['sex','diabetes_status'], observed=False)['target_michd'].mean().reset_index().dropna()
plt.figure(figsize=(8,4))
sns.barplot(data=combo, x='diabetes_status', y='target_michd', hue='sex', palette='Set2')
plt.title('Attribute Combination: Prevalence by Sex and Diabetes Status')
plt.ylabel('Prevalence')
plt.xlabel('Diabetes Status')
plt.xticks()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Prepare the data for machine learning algorithms

# %%

num_df = clean_df[NUMERIC_FEATURES].copy()
medians = {c: float(num_df[c].median()) for c in NUMERIC_FEATURES}
means = {c: float(num_df[c].fillna(medians[c]).mean()) for c in NUMERIC_FEATURES}
stds = {c: float(num_df[c].fillna(medians[c]).std(ddof=0)) if float(num_df[c].fillna(medians[c]).std(ddof=0)) != 0 else 1.0 for c in NUMERIC_FEATURES}
num_df = num_df.fillna(medians)
for c in NUMERIC_FEATURES:
    num_df[c] = ((num_df[c] - means[c]) / stds[c]).astype('float32')

cat_values = {
    'sex': ['Male','Female'],
    'age_group': ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+'],
    'race_ethnicity': ['White_non_Hispanic','Black_non_Hispanic','Other_non_Hispanic','Multiracial_non_Hispanic','Hispanic'],
    'income_group': ['lt_15k','15k_25k','25k_35k','35k_50k','50k_100k','100k_200k','ge_200k'],
    'education_group': ['No_HS_Diploma','HS_Graduate','Some_College','College_Graduate'],
    'marital_status': ['Married','Divorced','Widowed','Separated','Never_Married','Unmarried_Couple'],
    'employment_status': ['Employed_for_wages','Self_employed','Out_of_work_gt_1yr','Out_of_work_lt_1yr','Homemaker','Student','Retired','Unable_to_work'],
    'general_health': ['Excellent','Very_Good','Good','Fair','Poor'],
    'couldnt_afford_doctor': ['Yes','No'],
    'last_checkup': ['Within_past_year','Within_past_2_years','Within_past_5_years','Five_or_more_years_ago','Never'],
    'exercise_30d': ['Yes','No'],
    'diabetes_status': ['Yes','Yes_during_pregnancy','No','Prediabetes'],
    'ecig_status': ['Every_day','Some_days','Not_at_all','Never_used'],
    'smoker_status': ['Current_every_day','Current_some_days','Former_smoker','Never_smoked'],
    'physically_active': ['Active','Inactive'],
}

cat_df = clean_df[CATEGORICAL_FEATURES].copy()
for c in CATEGORICAL_FEATURES:
    mode = cat_df[c].mode(dropna=True)
    fill_val = mode.iloc[0] if len(mode) else cat_values[c][0]
    if fill_val not in cat_df[c].cat.categories:
        cat_df[c] = cat_df[c].cat.add_categories([fill_val])
    cat_df[c] = cat_df[c].fillna(fill_val)
    cat_df[c] = pd.Categorical(cat_df[c], categories=cat_values[c])

dummy_df = pd.get_dummies(cat_df, prefix_sep='=', dtype='float32')
processed_df = pd.concat([
    clean_df[['target_michd','sample_weight']].reset_index(drop=True),
    num_df.reset_index(drop=True),
    dummy_df.reset_index(drop=True)
], axis=1)
print(processed_df.shape)
processed_df.head()


# %% [markdown]
# ## 9. Dimensionality reduction (PCA experiment)

# %%
# Run PCA on 50,000 row sample to estimate dimensionality needed for 95% variance retention
pca_sample = processed_df.drop(columns=['target_michd','sample_weight']).sample(n=min(50000, len(processed_df)), random_state=42)
pca = PCA(random_state=42).fit(pca_sample)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n95 = int(np.argmax(cum_var >= 0.95) + 1)
print('Components needed for 95% variance:', n95)
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o', ms=3)
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.axvline(n95, color='green', linestyle='--')
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Run PCA on full dataset to estimate dimensionality needed for 95% variance retention
pca_sample = processed_df.drop(columns=['target_michd','sample_weight'])
pca = PCA(random_state=42).fit(pca_sample)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n95 = int(np.argmax(cum_var >= 0.95) + 1)
print('Components needed for 95% variance:', n95)
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o', ms=3)
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.axvline(n95, color='green', linestyle='--')
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 10. Export the deliverables

# %%

clean_path = OUT / 'brfss2024_heart_bias_clean_selected.csv'
ml_path = OUT / 'brfss2024_heart_bias_preprocessed_ml.csv'
clean_df.to_csv(clean_path, index=False)
processed_df.to_csv(ml_path, index=False)
print(clean_path)
print(ml_path)



