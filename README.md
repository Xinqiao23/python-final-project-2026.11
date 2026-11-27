# Task 2 – Heart Failure Analysis

Final project for Python for Data Science. We looked at records from 299 patients with heart failure to see which characteristics were associated with death during follow-up. We also compared three classification models on the same test set.

## Files

| File | Contents |
|---|---|
| [`heart_failure.ipynb`](heart_failure.ipynb) | Code, plots, and answers to Questions 1–8 |
| [`heart_failure.html`](heart_failure.html) | Notebook export that can be opened in a browser |
| [`heart_failure.csv`](heart_failure.csv) | Patient data used in the notebook |
| Report | Written project report, to be added |

## Run the notebook

From the `task 2` folder, install the dependencies and open the notebook.

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn notebook
jupyter notebook heart_failure.ipynb
```

Keep the CSV next to the notebook and run the cells from top to bottom.

## Results

- 96 of the 299 patients died during follow-up (32.1%). Of those 96 deaths, about 72% occurred within the first 90 days.
- Smoking was more common among men than women (47.4% vs. 3.8%). The overall comparison by smoking status showed little difference in mortality (p = 0.93).
- Patients who died tended to be older and have lower ejection fraction, higher serum creatinine, and lower serum sodium.

## Model comparison

The test set contained 60 patients, including 19 who died. The models use patient characteristics available before follow-up. The data splits and oversampling use `random_state=0`.

| Model | Test accuracy | Deaths identified |
|---|---:|---:|
| K-nearest neighbors (k = 7) | 66.7% | 3 of 19 |
| Logistic regression | 78.3% | 8 of 19 |
| Logistic regression with oversampling | 75.0% | 12 of 19 |

Logistic regression had the highest accuracy. Oversampling helped it identify four more deaths, with eight false alarms compared with two for the original model.
