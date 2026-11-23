# Task 2: Heart Failure Analysis

Python for Data Science – final project.

A hospital cardiology department wants to find the factors that predict deadly heart failure, in order to identify at-risk patients. This project explores a dataset of 299 heart failure patients and builds simple classification models to predict death during the follow-up period.

## Files

| File | Description |
|---|---|
| `heart_failure.ipynb` | Main Jupyter notebook: code, figures and interpretations for Q1–Q8 |
| `heart_failure.py` | The same code as a plain Python script (exported from the notebook) |
| `heart_failure.html` | HTML version of the notebook (opens in any web browser) |
| `heart_failure.csv` | The dataset (299 patients, 13 columns) |

All files must stay in the same folder, because the code reads the data with:

```python
heart = pd.read_csv("heart_failure.csv")
```

## Questions

1. What is the mortality rate?
2. How are the variables distributed?
3. How do the variables relate to each other?
4. Is the smoker distribution the same for both genders?
5. How many patients have each combination of underlying conditions?
6. Further analysis
7. What are the main risk factors?
8. (Optional) Classification model

## What the code does

| Part | Content |
|---|---|
| Data check | Load the data, check types, missing values (none) and duplicates (none) |
| Q1 | Mortality rate |
| Q2 | Distribution of binary variables (pie charts) and continuous variables (histograms) |
| Q3 | Correlation matrix (heatmap) |
| Q4 | Smoking by sex: cross table, smoking rate for each sex, chi-square test |
| Q5 | Number of patients for each combination of anaemia, diabetes and high blood pressure |
| Q6 | Death rate for each combination of sex and smoking; when deaths happen during the follow-up |
| Q7 | Risk factors: t-tests for continuous variables, chi-square tests for binary variables |
| Q8 | Classification models predicting death (see below) |

**Q8 models**
- Data split: training 70%, validation 10%, test 20% (stratified)
- Model 1: KNN (k chosen on the validation set)
- Model 2: Logistic regression
- Model 3: Logistic regression with oversampling of the deaths
- Comparison with accuracy, recall, precision, F1, missed deaths and false alarms

The variable `time` is not used in Q7 and Q8, because follow-up ends when a patient dies (it would leak the answer).

## Main findings

- The mortality rate is **32.1%** (96 of 299 patients).
- Men smoke much more than women (47.4% vs 3.8%), but smoking is **not** related to death.
- 72% of deaths happen within the **first 90 days** of follow-up.
- The main risk factors are **low ejection fraction, high serum creatinine and older age**, followed by low serum sodium.
- Logistic regression has the highest accuracy (78%); with oversampling it finds the most deaths (12 of 19 in the test set).

## Environment

The code was tested with:

| Package | Version |
|---|---|
| Python | 3.9.6 |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| matplotlib | 3.9.4 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |

Install the packages (Jupyter is only needed for the notebook):

```
pip install pandas numpy matplotlib scipy scikit-learn notebook
```

## How to run

**Option A – Jupyter notebook (recommended)**

1. Open a terminal in the folder that contains the files.
2. Run `jupyter notebook heart_failure.ipynb` (or open the file in VS Code and select a Python 3 kernel).
3. Click **Run All**. The cells must be run in order from top to bottom, because later questions use variables created earlier (for example `survived`, `died` and `mortality_rate`).

**Option B – Python script**

1. Open a terminal in the folder that contains the files.
2. Run `python3 heart_failure.py`.
3. Results are printed in the terminal. Each figure opens in a new window; close the window to let the script continue.

The whole analysis runs in a few seconds.

## Reproducibility

All random steps (the train/validation/test split and the oversampling) use a fixed seed (`random_state=0`), so running the code again gives exactly the same results.

Expected results on the test set (60 patients, 19 deaths):

| Model | Accuracy | Missed deaths |
|---|---|---|
| KNN (k=7) | 0.6667 | 16 |
| Logistic regression | 0.7833 | 11 |
| Logistic regression + oversampling | 0.7500 | 7 |

## Notes

- On some Mac computers, the logistic regression cells print `RuntimeWarning: divide by zero encountered in matmul`. This is a known numpy issue on macOS and does not affect the results. It can be hidden by adding these two lines after the imports:

  ```python
  import warnings
  warnings.filterwarnings('ignore')
  ```

- In the `.py` script, tables are printed with `print()`; in the notebook they are displayed automatically.
