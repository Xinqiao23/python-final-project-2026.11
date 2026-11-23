Task 2: Heart Failure Analysis - How to run the code
=====================================================

1. Files
--------
heart_failure.csv     The dataset (299 patients, 13 columns).
heart_failure.ipynb   Main Jupyter notebook: code, figures and interpretations for Q1-Q8.
heart_failure.py      The same code as a plain Python script (exported from the notebook).
heart_failure.html    HTML version of the notebook, can be opened in any web browser.

All files must stay in the same folder, because the code reads the data with:
    heart = pd.read_csv("heart_failure.csv")


2. Environment
--------------
The code was tested with:
    Python        3.9.6
    pandas        2.3.3
    numpy         2.0.2
    matplotlib    3.9.4
    scipy         1.13.1
    scikit-learn  1.6.1

To install the packages:
    pip install pandas numpy matplotlib scipy scikit-learn

Jupyter is also needed to open the notebook:
    pip install notebook


3. How to run
-------------
Option A - Jupyter notebook (recommended)
    1. Open a terminal in the folder that contains the files.
    2. Run:  jupyter notebook heart_failure.ipynb
       (or open the file in VS Code and select a Python 3 kernel)
    3. Click "Run All". The cells must be run in order from top to bottom,
       because later questions use variables created earlier
       (for example "survived", "died" and "mortality_rate").

Option B - Python script
    1. Open a terminal in the folder that contains the files.
    2. Run:  python3 heart_failure.py
    3. Results are printed in the terminal. Each figure opens in a new window;
       close the window to let the script continue.

The whole analysis runs in a few seconds.


4. What the code does
---------------------
Data check   Load the data, check types, missing values (none) and duplicates (none).
Q1           Mortality rate.
Q2           Distribution of binary variables (pie charts) and continuous variables (histograms).
Q3           Correlation matrix (heatmap).
Q4           Smoking by sex: cross table, smoking rate for each sex, chi-square test.
Q5           Number of patients for each combination of anaemia, diabetes and high blood pressure.
Q6           Further analysis: death rate for each combination of sex and smoking;
             when deaths happen during the follow-up.
Q7           Risk factors: t-tests for continuous variables, chi-square tests for binary variables.
Q8           Classification models predicting death:
               - data split: training 70%, validation 10%, test 20% (stratified)
               - Model 1: KNN (k chosen on the validation set)
               - Model 2: Logistic regression
               - Model 3: Logistic regression with oversampling of the deaths
               - comparison with accuracy, recall, precision, F1, missed deaths and false alarms
             The variable "time" is not used in Q7 and Q8, because follow-up ends when
             a patient dies (it would leak the answer).


5. Reproducibility
------------------
All random steps use a fixed seed (random_state=0): the train/validation/test split
and the oversampling. Running the code again gives exactly the same results.

Expected main results on the test set (60 patients, 19 deaths):
    KNN (k=7)                             accuracy 0.6667   missed deaths 16
    Logistic regression                   accuracy 0.7833   missed deaths 11
    Logistic regression + oversampling    accuracy 0.7500   missed deaths 7


6. Notes
--------
- On some Mac computers, the logistic regression cells print "RuntimeWarning: divide by
  zero encountered in matmul". This is a known numpy issue on macOS and does not affect
  the results. It can be hidden by adding these two lines after the imports:
      import warnings
      warnings.filterwarnings('ignore')
- In the .py script, tables are printed with print(); in the notebook they are displayed
  automatically.
