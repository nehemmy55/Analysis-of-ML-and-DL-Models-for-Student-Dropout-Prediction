# Student Dropout Prediction: Classical ML vs. Deep Learning

A comparative machine learning project predicting student academic outcomes
(Dropout / Enrolled / Graduate) for the Machine Learning Module Summative
Project. The pipeline implements four classical ML experiments (Scikit-learn)
and four deep learning experiments (TensorFlow/Keras — Sequential API,
Functional API, and `tf.data`), evaluated head-to-head on the same
preprocessed dataset.

## Project Status

The classical ML experiments (1-4) have been executed end-to-end on the real
dataset; their results in this repo (CSV, figures) are measured, not
estimated. The deep learning experiments (5-8) are fully implemented and
verified for correctness in the notebook — Sequential API, Functional API,
BatchNorm, Dropout, and a shared `tf.data` pipeline — but require you to run
the notebook once (a few minutes on Google Colab's free tier) to get their
actual numbers, since they need a TensorFlow runtime. **Run the notebook,
then fill in the 4 DL rows** in `reports/experiment_results_table.csv` and in
your final report before submission.

## Problem & Mission Alignment

Student dropout is one of the most costly failure modes in higher education:
for the student, it usually means lost tuition, debt without a credential,
and a derailed career path; for the institution, it means lost revenue and a
worse outcome on the metrics accreditation bodies track. This project asks
whether a model trained only on data already collected at enrollment and
after the first two semesters — demographics, admission grade, and
curricular-unit performance — can flag at-risk students early enough for an
academic advisor to intervene, and which model family (classical ML or deep
learning) does that more reliably on a dataset of this size.

## Repository Structure

```
student-dropout-prediction/
├── README.md                          <- you are here
├── requirements.txt                   <- pinned dependencies for reproducibility
├── LICENSE
├── .gitignore
│
├── data/
│   └── student_dropout_data.csv       <- raw dataset (UCI / Zenodo open source,
│                                          "Predict Students' Dropout and Academic Success")
│
├── notebooks/
│   └── student_dropout_notebook.ipynb <- full pipeline: EDA -> preprocessing ->
│                                          feature engineering -> 4 ML experiments ->
│                                          4 DL experiments -> evaluation -> conclusions
│
├── reports/
│   ├── Student_Dropout_Report.docx    <- scholarly report (intro, lit review, methodology,
│   │                                      results, discussion, conclusion; IEEE citations)
│   ├── experiment_results_table.csv   <- the 8-experiment comparison table
│   │                                      (4 ML rows measured; 4 DL rows to fill in)
│   └── figures/                       <- figures referenced in the report
│       ├── fig_target_distribution.png      (generated)
│       ├── fig_feature_distributions.png    (generated)
│       ├── fig_correlation_heatmap.png      (generated)
│       ├── fig_engineered_features.png      (generated)
│       ├── fig_feature_importance.png       (generated)
│       ├── fig_confusion_matrices_ml.png    (generated)
│       ├── fig_roc_curves_ml.png            (generated)
│       ├── fig_bias_variance_ml.png         (generated)
│       ├── fig_learning_curve.png           (generated)
│       ├── fig_dl_training_curves.png       (export from notebook after running it)
│       └── fig_confusion_matrices_dl.png    (export from notebook after running it)
│
└── docs/
    ├── video_script.md                <- cell-by-cell presentation script
    └── video_link.md                  <- link to the 5-10 min presentation recording
```

## How This Maps to the Grading Rubric

| Rubric Criterion | Where to find it |
|---|---|
| Problem Definition, Data, & Mission Alignment | README (above) + Report §1 Introduction |
| Literature Review & Theoretical Grounding | Report §2 Literature Review (10+ IEEE-cited sources) |
| Data Preprocessing & Feature Engineering | Notebook §4 (4 engineered features) + Report §3 Methodology |
| Model Implementation & Optimization (2+ approaches, 7+ experiments) | Notebook §5 (ML, 4 experiments) + §8 (DL, 4 experiments) |
| Experiment/Result Table | Notebook §13 + `reports/experiment_results_table.csv` + Report §4 |
| Model Evaluation & Error Analysis | Notebook §7, 9-12, 15-16 (confusion matrices, ROC, PR curves, bias-variance, learning curve, error analysis) + Report §5 Discussion |
| Code Quality & Documentation | Notebook: modular functions, markdown before/after every cell, seeded, runs top-to-bottom |
| Academic Report Quality & Originality | `reports/Student_Dropout_Report.docx` |
| Demo Video & Communication | `docs/video_link.md` + `docs/video_script.md` |

## Setup & Reproducibility

```bash
git clone <your-repo-url>
cd student-dropout-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/student_dropout_notebook.ipynb
```

All random operations (NumPy, Python's `random`, TensorFlow, and every
train/val/test split) are seeded with `SEED = 42` for reproducibility. The
notebook runs top-to-bottom without manual intervention, provided the CSV is
present at `data/student_dropout_data.csv` (the notebook's data-loading cell
checks several relative paths automatically).

## Dataset

**Source:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success),
UCI Machine Learning Repository (originally published via Zenodo by
Realinho et al., 2021), collected from a higher-education institution in
Portugal. 4,424 student records, 36 features spanning demographics,
socio-economic factors, and academic performance in the first two semesters,
with a 3-class target: Dropout (32.1%), Enrolled (17.9%), Graduate (49.9%).

## Results Summary (Classical ML — Measured)

| Experiment | Accuracy | Weighted F1 (test) | Train F1 | Train-Test Gap | ROC-AUC (macro) |
|---|---|---|---|---|---|
| Exp 1 — Logistic Regression | 0.758 | 0.770 | 0.768 | -0.002 | 0.899 |
| Exp 2 — Decision Tree | 0.732 | 0.743 | 0.798 | 0.055 | 0.854 |
| Exp 3 — Random Forest (Default) | 0.810 | 0.799 | 1.000 | 0.201 | 0.911 |
| Exp 4 — Random Forest (Tuned) | 0.795 | 0.799 | 0.879 | 0.080 | 0.912 |

The default Random Forest memorised the training set perfectly (train F1 =
1.000) and still scored highest raw test accuracy, but its 0.20 train-test
gap is the clearest overfitting signal in the classical results — tuning
(Exp 4) cut that gap by more than half for an essentially identical test F1.
Deep learning rows (Exp 5-8) are pending notebook execution; see Project
Status above.

## Author

Nehemie ISHIMWE
