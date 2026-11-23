# Task 2: Heart Failure Analysis
# (exported from heart_failure.ipynb)

# ----------------------------------------------------------------------
# # Task 2: Heart Failure Analysis
#
# Questions
# 1. What is the mortality rate due to heart failure among the study's participants?
# 2. How are the different variables (age, gender, presence of diabetes or high blood pressure, etc.) distributed in the dataset?
# 3. How do the variables relate to each other?
# 4. Is the smoker distribution the same for both genders?
# 5. How many patients present each combination of underlying health conditions (anaemia only, diabetes only, anaemia + diabetes, etc.)?
# 6. etc.
# 7. What are the main factors that are related to a higher risk of heart failure?
# 8. (Optional) Build a simple machine learning classification model that predicts if a patient is at risk of dying from heart failure.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# show all columns when printing tables in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# read and check data
heart = pd.read_csv("heart_failure.csv")
print(heart.head())

print(f"rows: {heart.shape[0]}")
print(f"columns: {heart.shape[1]}")

heart.info()

# round to 2 dicimals
print(heart.describe().round(2))

# check missing value
for column in heart.columns:
    missing_ratio = heart[column].isna().mean()
    print(f"{column} missing ratio: {missing_ratio:}")

# ----------------------------------------------------------------------
# There's no missing value so don't need to do replacement.

# binary varialbles
binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# continuous variables
continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# classification by died or not
survived = heart[heart['DEATH_EVENT'] == 0]
died = heart[heart['DEATH_EVENT'] == 1]

# ----------------------------------------------------------------------
# ## Q1. What is the mortality rate?
#
# mortality_rate = death_count / total

total = heart.shape[0]
death_count = heart['DEATH_EVENT'].sum()
survived_count = total - death_count
mortality_rate = death_count / total
survival_rate = survived_count / total
print(f"Total patients: {total}")
print(f"Died: {death_count}")
print(f"Survived: {survived_count}")
print(f"Mortality rate: {mortality_rate:.2%}")
print(f"Survival rate: {survival_rate:.2%}")

# ----------------------------------------------------------------------
# The mortality rate is 32.11%.

# ----------------------------------------------------------------------
# ## Q2. How are the variables distributed?
#
# ### 2.1 Binary variables

# rename binary variable
binary_labels = {
    'anaemia': ['No', 'Yes'],
    'diabetes': ['No', 'Yes'],
    'high_blood_pressure': ['No', 'Yes'],
    'sex': ['Female', 'Male'],
    'smoking': ['No', 'Yes']
}

# pie charts for binary variables: 3 plots per row
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()    # make those subplots into a row

for i in range(len(binary_cols)):
    col = binary_cols[i]
    count_1 = heart[col].sum()       # number of yes
    count_0 = total - count_1        # number of no
    percent_0 = count_0 / total        # percent of no
    percent_1 = count_1 / total        # percent of yes

    label_0 = f"{binary_labels[col][0]}\n{count_0} ({percent_0:.1%})"
    label_1 = f"{binary_labels[col][1]}\n{count_1} ({percent_1:.1%})"

    axes[i].pie([count_0, count_1], labels=[label_0, label_1],
                colors=['steelblue', 'orange'], startangle=90)
    axes[i].set_title(col)

# delete last one
fig.delaxes(axes[5])

plt.suptitle('Distribution of binary variables')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# ### 2.2 Continuous variables

# 3*3
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()    # make those subplots into a row
for i in range(len(continuous_cols)):
    col = continuous_cols[i]
    axes[i].hist(heart[col], bins=20, color='grey', edgecolor='white')
    axes[i].axvline(heart[col].mean(), color='red', linestyle='--', label='mean')
    axes[i].axvline(heart[col].median(), color='yellow', linestyle='--', label='median')
    axes[i].set_title(col)
    axes[i].legend()

# delete last two empty subplots
fig.delaxes(axes[7])
fig.delaxes(axes[8])

plt.suptitle(' Distribution of continuous variables')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# binary：
# - male65%, female35%。
# - anaemia 43%，diabetes42%，high blood pressure35%，smoking 32%
#
# continuous：
# - **age**：most are older people
# - **ejection_fraction**：about 38%, normal people 50%-70%，so most patients' ejection fraction decreases
# - **creatinine_phosphokinase** and **serum_creatinine**：right skrewed
# - **serum_sodium**：little left skrewed
# - **platelets**：symmetry
# - **time**：from 4 to 285, very spread out

# ----------------------------------------------------------------------
# ## Q3. How do the variables relate to each other?
#
# calculate confusion matrix and draw heatmap

corr = heart.corr(numeric_only=True)

plt.figure(figsize=(11, 9))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)    # red: positive correlation, blue: negative correlation
plt.colorbar()


plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

# heatmap and correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)

plt.title('Correlation matrix')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Most variables are only weakly correlated with each other
# the strongest link between features is sex and smoking (r = 0.45), meaning men are much more likely to smoke.
#
# Death is positively correlated with serum creatinine (r = 0.29) and age (r = 0.25), and negatively correlated with ejection fraction (r = -0.27) and serum sodium (r = -0.20).
#
# Time shows the strongest correlation with death (r = -0.53), but this is because follow-up ends when a patient dies, so it is a consequence not a risk factor.

# ----------------------------------------------------------------------
# ## Q4. Is the smoker distribution the same for both genders?

# do a cross table
smoke_sex = pd.crosstab(heart['sex'], heart['smoking'])

# change names
smoke_sex.index = ['Female', 'Male']
smoke_sex.columns = ['Non-smoker', 'Smoker']
print(smoke_sex)

# calculate rate of smoker for each gender
sex_names = ['Female', 'Male']

for sex_value in [0, 1]:                                  # 0 = Female, 1 = Male
    group = heart[heart['sex'] == sex_value]              
    nb_total = len(group)                                 # total number of people of this gender
    nb_smoker = group['smoking'].sum()                    # number of smokers (0/1 sum)
    percent_smoker = nb_smoker / nb_total

    print(f"{sex_names[sex_value]}: {nb_smoker} smokers out of {nb_total} ({percent_smoker:.1%})")

# ----------------------------------------------------------------------
# ### Chi-square test
#
# The chi-square test  is used to test whether two categorical variables are independent:
# - **H0**: Smoking and sex are not related.
# - **H1 (alternative hypothesis)**: Smoking and sex are related.
# - If the **p-value < 0.05**, reject H0, so the proportion of smokers is significantly different between men and women.

chi2, p_value, dof, expected = stats.chi2_contingency(smoke_sex)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"p-value: {p_value:.2e}")    

if p_value < 0.05:
    print("smoking distribution is different between genders")
else:
    print("no significant difference")

# ----------------------------------------------------------------------
# ## Q5. How many patients have each combination of underlying conditions?
#
# 3 underlying conditions：anaemia , diabetes , high_blood_pressure.
# total 8 combinations

def condition_label(row):
    '''
    This function takes one patient and returns
    the combination of underlying conditions as a string.
    '''
    conditions = []
    if row['anaemia'] == 1:
        conditions.append('Anaemia')
    if row['diabetes'] == 1:
        conditions.append('Diabetes')
    if row['high_blood_pressure'] == 1:
        conditions.append('High BP')

    if len(conditions) == 0:
        return 'None'
    else:
        return ' and '.join(conditions)

# give each patient a label for their underlying conditions
label_list = []
for i, row in heart.iterrows():    
    label_list.append(condition_label(row))

heart['conditions'] = label_list
print(heart[['conditions']].head(10))

# count
combo_counts = heart['conditions'].value_counts()

for combo in combo_counts.index:
    count = combo_counts[combo]      # count of patients with this combination
    percent = count / total          # proportion 
    print(f"{combo}: {count} patients ({percent:.1%})")

# ----------------------------------------------------------------------
# Only about 22% of patients have none of the three conditions, so nearly 4 out of 5 heart failure patients have at least one underlying condition.

# ----------------------------------------------------------------------
# ## Q6. Further analysis
#
# This section contains two additional analyses, each with a clear conclusion:
# 1. **Is smoking related to death?** (A follow-up to Q4: men smoke much more than women, so does smoking affect death?)
# 2. **When do deaths happen?**

# ----------------------------------------------------------------------
# ### 6.1 Smoking, sex and death
#
# Q4 showed that the proportion of smokers is much higher among men than among women (3.8%). So if we simply compare smokers and non-smokers, the effect of smoking and the effect of sex are mixed together.
#
# so we stratify by sex.
#
# We first look at all patients, then at women and men separately. If smoking really affects death, we should also see a difference within the same sex.
# For each group, a chi-square test is used to check whether the difference is significant (p < 0.05).

# death rate for each combination of sex and smoking
group_names = []
death_rates = []
group_n = []

for sex_value in [0, 1]:
    for smoke_value in [0, 1]:
        group = heart[(heart['sex'] == sex_value) & (heart['smoking'] == smoke_value)]

        sex_name = ['Female', 'Male'][sex_value]
        smoke_name = ['Non-smoker', 'Smoker'][smoke_value]
        group_names.append(f"{sex_name}\n{smoke_name}")

        death_rates.append(group['DEATH_EVENT'].mean())
        group_n.append(len(group))

plt.figure(figsize=(8, 6))
plt.bar(group_names, death_rates, color=['orchid', 'purple', 'lightskyblue', 'steelblue'])
plt.axhline(mortality_rate, color='black', linestyle='--', label='overall mortality')

for i in range(len(group_names)):
    plt.text(i, death_rates[i] + 0.02, f"{death_rates[i]:.0%}\n(n={group_n[i]})", ha='center')

plt.ylabel('Death rate')
plt.ylim(0, 1)
plt.title('Q6: Death rate by sex and smoking')
plt.legend()
plt.show()

# ----------------------------------------------------------------------
# **All patients**: smokers and non-smokers have almost the same death rate (31% vs 33%, p = 0.93).
# **Men**: the death rates are also similar (29% vs 34%, p = 0.56).
# **Women**: the number was large at 75%, but there are only 4 smokers, too few to draw a conclusion.
#
# So smoking is not a risk factor for death, even when we look at men and women separately.

# ----------------------------------------------------------------------
# ### 6.2 When do deaths happen?
#
# time is the follow-up period in days. For patients who died, time is **the day of follow-up on which they died**.
# We look only at the 96 patients who died and count how many of them died within the first 30, 60 and 90 days.

# when do deaths happen?
for days in [30, 60, 90]:
    nb_early = (died['time'] <= days).sum()
    percent_early = nb_early / len(died)
    print(f"Deaths within {days} days: {nb_early} / {len(died)} ({percent_early:.1%})")

print(f"\nMedian follow-up time (Survived): {survived['time'].median()} days")
print(f"Median follow-up time (Died): {died['time'].median()} days")

# compare follow-up time of survived and died patients
plt.figure(figsize=(8, 5))
plt.hist(survived['time'], bins=20, alpha=0.6, color='steelblue', label='Survived')
plt.hist(died['time'], bins=20, alpha=0.6, color='indianred', label='Died')
plt.axvline(90, color='black', linestyle='--', label='90 days')
plt.xlabel('Follow-up time (days)')
plt.ylabel('Number of patients')
plt.title('Q6: Follow-up time by death event')
plt.legend()
plt.show()

# ----------------------------------------------------------------------
#
# Most deaths happen early: 72% of deaths occur within the first 90 days
# So patients should be closely monitored in the first three months.
#
# And from this we can also conclude that `time` is not a risk factor. Because follow-up ends when a patient dies,  a short follow-up time is a result of death, not a cause.

# ----------------------------------------------------------------------
# ## Q7. What are the main factors associated with a higher risk?
#
# We use statistical test to check whether the differences between the survived group and the died group are significant.
#
# **Continuous variables: t-test**
# H0: the survived group and the died group have the same mean
#
# If p < 0.05, the means of the two groups are significantly different, so this variable is related to death
#
# **Binary variables: chi-square test**
# H0: this variable is not related to death
#
# If p < 0.05, it is related to death

# continuous variables: compare survived and died patients
cont_rows = []

for col in continuous_cols:
    if col == 'time':    # go across time
        continue

    t_stat, p_value = stats.ttest_ind(survived[col], died[col], equal_var=False)    # equal_var=False 就是 Welch's t-test

    row = {
        'variable': col,
        'survived_mean (SD)': f"{survived[col].mean():.2f} ({survived[col].std():.2f})",
        'died_mean (SD)': f"{died[col].mean():.2f} ({died[col].std():.2f})",
        'p_value': round(p_value, 4),
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    cont_rows.append(row)

cont_result = pd.DataFrame(cont_rows).sort_values('p_value')
print(cont_result)

# binary variables: compare survived and died patients
bin_rows = []

for col in binary_cols:
    table = pd.crosstab(heart[col], heart['DEATH_EVENT'])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)

    death_rate_0 = heart[heart[col] == 0]['DEATH_EVENT'].mean()
    death_rate_1 = heart[heart[col] == 1]['DEATH_EVENT'].mean()

    row = {
        'variable': col,
        'death_rate if 0': round(death_rate_0, 3),
        'death_rate if 1': round(death_rate_1, 3),
        'p_value': round(p_value, 4),
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    bin_rows.append(row)

bin_result = pd.DataFrame(bin_rows).sort_values('p_value')
print(bin_result)

# ----------------------------------------------------------------------
#
# **Significant factors** (p < 0.05): patients who died had a lower ejection fraction, higher serum creatinine, were older and had lower serum sodium.
#
# **Not significant**: creatinine phosphokinase, platelets, anaemia, diabetes, high blood pressure, sex and smoking.
#
# So the main risk factors are low ejection fraction, high serum creatinine and older age, followed by low serum sodium. in conclusion, heart function, kidney function and age matter most.

# ----------------------------------------------------------------------
# ## Q8 (Optional). Classification model
#
# **Goal**: predict whether a patient will die from their features.
#
# **Models**:
# 1. KNN
# 2. Decision tree
# 3. Logistic regression
# 4. Logistic regression with oversampling
#
#
#  `time` is not used, and neither is the `conditions` column created in Q5.

# select features
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
target = 'DEATH_EVENT'

X = heart[features]
Y = heart[target]

print(f"X shape: {X.shape}")
print(Y.value_counts())

# learning set (80%) and test set (20%)
X_learn, X_test, Y_learn, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0, stratify=Y   # stratified sampling keeps the same death rate (about 32%) in each set. 
)

# learning set :training set (70%) and validation set (10%)
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_learn, Y_learn, test_size=0.125, random_state=0, stratify=Y_learn
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_valid.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

baseline_accuracy = (Y_test == 0).mean()
print(f"Baseline accuracy (always predict 'Survived'): {baseline_accuracy:.4f}")

# ----------------------------------------------------------------------
# ### Model 1: KNN
#
# **Method**: to predict a new patient, KNN finds the k most similar patients in the training set and takes the majority class.
#
# **Why**:
# - It is a simple model that makes no assumption about the data.
# - KNN uses distances, so the variables must be standardized first.
# - We only try odd values like 1, 3, ..., 31 to avoid ties, and choose k on the validation set.

knns = range(1, 33, 2)
knn_acc_train = []
knn_acc_valid = []

for k in knns:
    knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn_model.fit(X_train, Y_train)                                    # train on the training set
    acc_train = accuracy_score(Y_train, knn_model.predict(X_train))
    acc_valid = accuracy_score(Y_valid, knn_model.predict(X_valid))    # evaluate on the validation set
    knn_acc_train.append(acc_train)
    knn_acc_valid.append(acc_valid)
    print(f"k={k}: train accuracy = {acc_train:.4f}, validation accuracy = {acc_valid:.4f}")

# the best k 
best_k = knns[np.argmax(knn_acc_valid)]
print(f"Best k: {best_k}")

plt.figure(figsize=(8, 5))
plt.plot(knns, knn_acc_train, 'bo-', label='training')
plt.plot(knns, knn_acc_valid, 'go-', label='validation')
plt.axhline(baseline_accuracy, color='red', linestyle='--', label='baseline')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN accuracy for different k')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# We retrain the model with the best k on the learning set and evaluate it once on the test set.
#
# **Evaluation metrics** (computed from the confusion matrix):
#  **Accuracy** = correct predictions / total
#  **Recall** = TP / (TP + FN)
#  **Precision** = TP / (TP + FP)

def evaluate(Y_true, Y_pred):
    '''
    This function prints the confusion matrix, accuracy, recall and precision,
    and returns the confusion matrix.
    '''
    cm = confusion_matrix(Y_true, Y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    print("Confusion Matrix:")
    print(cm)
    print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    print(f"Accuracy: {(tp + tn) / cm.sum():.4f}")
    print(f"Recall: {tp / (tp + fn):.4f}")
    print(f"Precision: {tp / (tp + fp):.4f}")
    return cm

best_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))
best_knn.fit(X_learn, Y_learn)
Y_predicted_knn = best_knn.predict(X_test)

confusion_mattrix_knn = evaluate(Y_test, Y_predicted_knn)

# ----------------------------------------------------------------------
# ### Model 2: Logistic regression
#
# **Method**: logistic regression estimates the probability of death. If the probability is above 0.5, the patient is predicted to die.
#
# Its coefficients show whether each variable increases or decreases the risk of death.
#
# The features are standardized first, so that the coefficients can be compared directly.

# standardize: fit the scaler on the learning set only, then apply it to the test set
scaler = StandardScaler()
X_learn_scaled = scaler.fit_transform(X_learn)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_learn_scaled, Y_learn)
Y_predicted_log = log_model.predict(X_test_scaled)

confusion_mattrix_log = evaluate(Y_test, Y_predicted_log)

# coefficients: positive = higher risk (red), negative = lower risk (blue)
coef_table = pd.DataFrame({'feature': features, 'coefficient': log_model.coef_[0]})
coef_table = coef_table.sort_values('coefficient')

colors = []
for c in coef_table['coefficient']:
    if c > 0:
        colors.append('indianred')
    else:
        colors.append('steelblue')

plt.figure(figsize=(8, 6))
plt.barh(coef_table['feature'], coef_table['coefficient'], color=colors)
plt.axvline(0, color='black')
plt.xlabel('Coefficient (standardized)')
plt.title('Logistic regression coefficients')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# **Interpretation**
#
# - On the test set, accuracy is 78%, the highest of the three models, and precision is 80% (8 of the 10 predicted deaths really died).
# - But only 8 of the 19 deaths are found (recall 42%): few false alarms, but many missed deaths.
# - The largest coefficients are ejection fraction (negative), serum creatinine (positive) and age (positive), which matches Q7.

# ----------------------------------------------------------------------
# ### Model 3: Logistic regression with oversampling
#
# **Method**
#
# We randomly duplicated records of patients who died in training set until both groups were the same size. Then retrained the same logistic regression model.
#
# **Why**
#
# The earlier models missed more than half of the deaths.
#
# There were twice as many survivors as deaths, so the models tended to predict survival.
#
# Balancing the training set can help the model detect more deaths.
#
# We kept the original class balance in the test set for a realistic evaluation.

# put X and Y together to split the learning set into survived and died
learn = X_learn.copy()
learn['DEATH_EVENT'] = Y_learn

learn_survived = learn[learn['DEATH_EVENT'] == 0]
learn_died = learn[learn['DEATH_EVENT'] == 1]
print(f"Before oversampling: survived = {len(learn_survived)}, died = {len(learn_died)}")

# randomly copy patients who died 
nb_extra = len(learn_survived) - len(learn_died)
extra_died = learn_died.sample(n=nb_extra, replace=True, random_state=0)

learn_over = pd.concat([learn_survived, learn_died, extra_died])
X_learn_over = learn_over[features]
Y_learn_over = learn_over['DEATH_EVENT']

print(f"After oversampling: survived = {(Y_learn_over == 0).sum()}, died = {(Y_learn_over == 1).sum()}")

# train on the oversampled data
X_learn_over_scaled = scaler.transform(X_learn_over)

log_over_model = LogisticRegression(max_iter=1000)
log_over_model.fit(X_learn_over_scaled, Y_learn_over)
Y_predict_log_over = log_over_model.predict(X_test_scaled)    # evaluated on the original test set

confusion_mattrix_log_over = evaluate(Y_test, Y_predict_log_over)

# ----------------------------------------------------------------------
# ### Model comparison
#
# Besides accuracy, we look at recall, F1 and missed deaths (FN), because missing a patient who will die is the most serious error.
#
# F1 = 2 × precision × recall / (precision + recall): it is high only if both precision and recall are high.
#
# the baseline (always predicting "survived") has an accuracy of 68.3% .

# compute the metrics of each model from its confusion matrix
models = [
    [f'KNN (k={best_k})', confusion_mattrix_knn],
    ['Logistic regression', confusion_mattrix_log],
    ['Logistic regression + oversampling', confusion_mattrix_log_over]
]

rows = []
for name, cm in models:
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    accuracy = (tp + tn) / cm.sum()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    rows.append({'model': name, 'accuracy': round(accuracy, 4), 'recall': round(recall, 4),
                 'precision': round(precision, 4), 'F1': round(f1, 4),
                 'missed deaths (FN)': fn, 'false alarms (FP)': fp})

compare = pd.DataFrame(rows)
print(compare)

# ----------------------------------------------------------------------
# KNN is worse than the baseline.
#
# Logistic regression has the highest accuracy (78%) but missed 11 of 19 deaths.
#
# Logistic regression with oversampling was the best model for screening. It has the highest F1 score, misses the fewest deaths, and has 75% accuracy.

