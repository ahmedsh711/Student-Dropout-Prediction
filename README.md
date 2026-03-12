# Student Dropout Prediction
**CSAI 801 — Winter 2026 | Queen's University | Group 8**

---

## Contributors

- Rodina Mohamed 
  [Github](https://github.com/Rodina222)

---

## Live App

🔗 [student-dropout-predictor-group8.streamlit.app](https://student-dropout-predictor-group8.streamlit.app)

---

## What is this?

Every year, a significant number of students leave university without finishing their degree. This project tries to predict — early enough to actually help — whether a student will **drop out**, stay **enrolled**, or **graduate**.

We built a machine learning pipeline trained on 4,349 students from a Portuguese polytechnic institute. Given a student's academic performance, financial situation, and personal background, the model outputs one of three outcomes and a confidence score for each.

---

## The Data

- **Source:** Realinho et al. (2021) — publicly available on UCI / Kaggle
- **Students:** 4,349 (after removing 75 data quality errors)
- **Features:** 22 selected from an original 35
- **Target classes:** Dropout (32%) · Enrolled (18%) · Graduate (50%)

The features cover three areas: academic performance (units approved, grades), financial status (tuition, scholarship, debt), and personal background (age, course, application order).

---

## How it works

**1. Clean the data** — removed 75 students labeled "Graduate" with zero approved units in both semesters. That's a data entry error, not an edge case.

**2. Engineer features** — averaged the 12 correlated semester columns into 6 cleaner ones, then added four paper-grounded features:
- `approval_rate` — units approved ÷ units enrolled (efficiency, not just volume)
- `sem2_approved_raw` — 2nd semester approvals kept separate, because averaging hides a crash in sem2
- `grade_trend` — sem2 grade minus sem1 grade, captures direction not just level
- `failed_units` — enrolled minus approved, captures the effort-vs-outcome gap

**3. Select features statistically** — Chi-Square + Cramér's V for categoricals, Spearman correlation for numericals. Only kept features with a statistically significant relationship to the outcome.

**4. Train and tune** — trained 12 baseline models (Logistic Regression, Decision Tree, Random Forest, SVM, MLP, XGBoost, LightGBM, CatBoost, AdaBoost, KNN, and two Dummy baselines), then tuned the top 7 using RandomizedSearchCV with 80 iterations and 5-fold cross-validation optimizing for F1-macro. SMOTE is applied inside each fold via ImbPipeline so validation folds never see synthetic samples.

**5. Calibrate the Enrolled threshold** — the default 0.5 cutoff is too conservative for the smallest class. We sweep thresholds down and pick the one that maximizes Enrolled F1 without losing more than 0.01 on overall F1-macro.

---

## Results

| Model | F1-macro | Balanced Acc | AUC |
|---|---|---|---|
| **Random Forest (Tuned)** | **0.7392** | **0.7386** | **0.8942** |
| LightGBM (Tuned) | 0.7383 | 0.7369 | 0.8956 |
| Soft Voting (Ensemble) | 0.7317 | 0.7341 | 0.8987 |
| Stacking (LR meta) | 0.7281 | 0.7385 | 0.8968 |

Per-class breakdown for the winning model:

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Dropout | 0.79 | 0.86 | 0.73 |
| Enrolled | 0.56 | 0.53 | 0.59 |
| Graduate | 0.87 | 0.85 | 0.89 |

Enrolled is the hardest class and always will be — 18% of the data, features that overlap with both other classes. The 0.56 F1 is close to the ceiling with semester-aggregated features.

---

## Why F1-macro?

Because we care equally about all three classes. Weighted F1 would let a model ignore Enrolled students (only 18% of data) and still look good on paper. Macro F1 treats each class equally — if you fail on Enrolled, the score reflects that.

---

## What we tried that didn't work

**SMOTE before cross-validation** — applying SMOTE to the full training set before CV caused synthetic samples to leak into validation folds, inflating CV scores to ~0.84 while test performance sat at ~0.70. Fixed by moving SMOTE inside each fold using ImbPipeline.

**Optuna Bayesian optimization** — smarter search, same ceiling. RF with Optuna scored 0.7248 vs 0.7267 with RandomizedSearch. The bottleneck is the data, not the optimizer.

**Ensemble beating individual model** — ensembles had better AUC (probability calibration) but slightly lower F1-macro than the tuned Random Forest alone.

---

## Project structure

```
├── preprocess.py     # data cleaning, feature engineering, feature selection, scaling
├── train.py          # tunes Random Forest, calibrates threshold, saves artifacts
├── streamlit_app.py  # web app
├── dataset.csv       # raw data
├── model.pkl         # trained Random Forest
├── scaler.pkl        # fitted StandardScaler
├── features.pkl      # selected feature list
└── threshold.pkl     # calibrated Enrolled threshold
```

---

## Run locally

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost shap streamlit joblib

python preprocess.py
python train.py
streamlit run streamlit_app.py
```

`preprocess.py` must run first — it generates the `preprocessed/` folder that `train.py` depends on.

---

## References

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). Predicting student dropout and academic success. *Data*, 7(11), 146.

Realinho, V. et al. (2022). Predicting student dropout and academic success. *Proc. DSAA 2022*.

Martins, M. V. et al. (2024). Financial factors in student dropout prediction. *Applied Sciences*.
