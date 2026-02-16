# DS.v2.5.3.4.1
## Machine Learning Capstone Project - Home Credit Default Risk Competition

<img src=reports/figures/app_figure.jpeg width=400 height=200> 
<br> 

This repository contains an end-to-end analysis of **Home Credit Default Risk** dataset. The goal is to build a robust **predictive** machine learning model using advanced feature engineering and multiple algorithms, and deploy it as a FastAPI application in a Docker container on Google Could Platform. 

#### Project overview
This project begins with a structured investigation plan, outlining the assumptions, goals, and proof-of-concept (POC) steps necessary for developing an ML product. The core analysis focuses on the Home Credit Default Risk dataset, where extensive exploratory data analysis (EDA) was conducted. To mimic a real-world scenario, 80% of the application_train dataset was used for model development, while the remaining 20% served as a truly unseen hold-out set for final generalization testing.

All auxiliary tables were explored in full to identify anomalies and understand patterns of missingness. Multiple feature selection approaches—including BorutaSHAP, correlation analysis, and domain-inspired grouping of predictors were used to identify the most influential features. Statistical inference techniques were applied to understand predictors’ relationships with the target and identify feature interactions with potential predictive value, also identify redundant features.

Several machine learning models were explored, with a focus on gradient boosting algorithms (XGBoost, LightGBM, HistGradientBoosting, CatBoost). Logistic Regression was included as a baseline model. Due to the dataset’s nature where missing values carry meaningful information tree-based models became the primary models of interest. Various feature engineering techniques were applied, including interaction terms, ratios, and domain-informed aggregations. Boruta-based feature selection was repeated at different stages to refine the predictor set.

The primary optimization metric was ROC AUC, since the dataset contains only ~8% positive defaulting cases. To address imbalance, both inner class weighting and scaling weight strategies were evaluated. Experiments were conducted to compare different preprocessing choices: imputing vs. leaving missing values, balance handling inside the model vs. external scaling, and alternative feature subsets. 

Hyperparameter tuning was performed using cross-validation and Bayesian optimization with Optuna. The final models (XGBoost, LightGBM, CatBoost, and HistGradientBoosting) were tuned thoroughly, then evaluated on the 20% hold-out set. SHAP values were used to analyze feature importance and model interpretability, and ROC AUC curves were generated for comparison.

Finally, Voting and Stacking ensembles were built and evaluated on the hold-out test set. All models were also submitted to Kaggle to assess generalization performance on the competition’s final test set. The best-performing model based on both internal evaluation and Kaggle ROC AUC was deployed as a FastAPI application, containerized with Docker, and deployed to Google Cloud Platform.

#### Dataset
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) is taken from *Kaggle*.  

The main application_train table contains ~308k samples with applicant-level information, supplemented by multiple relational tables:
* application_{train|test}.csv
* bureau.csv
* bureau_balance.csv
* POS_CASH_balance.csv
* credit_card_balance.csv
* previous_application.csv
* installments_payments.csv
* HomeCredit_columns_description.csv

Features include both categorical and numerical fields, many with domain-specific semantics.

 #### Tools and libraries
 This project uses:
* **numpy** - matrix manipulation and numerical computations.
* **pandas** - data manipulation and cleaning.
* **matplotlib, seaborn** - data visualization.
* **statsmodels, sklearn** -  ML models, statistical tests and metrics.
* **optuna** - Bayesian hyperparameter optimization.
* **borutaSHAP** -  feature selection.
* **xgboost, catboost, lightgbm** -  gradient boosted tree algorithms.
* **phik** - correlation analysis.
* **shap** - model explainability.
* **fastapi, docker** - model deployment and containerization.

#### Repository structure

Repository tree is as follows:
```
├── 341.md
├── Dockerfile
├── ml_app
│   ├── __init__.py
│   ├── app_utils
│   └── main.py
├── notebooks
│   ├── credits.ipynb
│   ├── files.ipynb
│   ├── applications.ipynb
│   ├── model_params
│   └── prev_application.ipynb
├── poetry.lock
├── reports
├── pyproject.toml
├── .gitignore
├── README.md
└── utils
    ├── transformers.py
    ├── utils.py
    └── plot_style.py
```
* Notebooks folder contains all analysis notebooks. applications.ipynb is the main notebook, others are helper files.
* ML_app folder contains FastAPI application code and saved model.
* Utils folder contains utility functions for plotting styles.

#### Overview of Models' Performance

This plot shows learning and generalization performance of the built models using ROC AUC score.   
<br>
<img src=reports/figures/final_models_comparison_roc_auc.png width=600 height=310> 
<br>

#### Key insights:
There are several key takeaways from the Home Credit Default Risk project:
1. **Data Understanding and Preprocessing**: Comprehensive exploratory data analysis (EDA) and preprocessing steps were crucial in understanding the datasets, handling missing values, outliers, and feature engineering. This laid a solid foundation for effective modeling. It was noticed that too strict winsorization of highly skewed features which were aggregated later led to loss of important information and reduced model learning performance, so a balanced approach was taken.

2. **Feature Selection**: The use of Boruta SHAP feature selection helped identify the most relevant features, reducing dimensionality and improving model interpretability without sacrificing predictive power. 

3. **Statistical Inference**: Conducting statistical tests validated the significance of selected features and informed feature interaction engineering, enhancing model robustness. Also, this enabled to identify a few redundant features making no harm to the predictive power of the models.

4. **Modeling Approach**: Gradient boosting models (LightGBM, XGBoost, CatBoost, HistGradientBoosting) were chosen for their ability to handle missing data, mixed data types, imbalanced datasets, and complex relationships. Hyperparameter tuning using Optuna further enhanced model performance. Logistic Regression was used as a baseline model to benchmark performance.

5. **Model Evaluation**: Rigorous evaluation using cross-validation and hold-out test sets ensured that the models generalized well to unseen data. The final test on the test subset proved that the models are stable and robust. The XGBoost model provided the best performance with a ROC AUC score of 0.772 on the final test set, compared to baseline Logistic Regression model ROC AUC score of 0.746. However, generalization performance on the hold-out test set was slightly better than on the cross-validated train set, indicating that the hold-out test set had slightly easier samples to predict, but lower than generalization performance on the final test set from Kaggle platform.

6. **Feature Importance**: Default risk seems to have the highest correlation wit EXT_SOURCE ratings, and credit card utilization ratios. Clients having greater annuity amount, shorter employment history, older car and more family members tend to have a higher risk of defaulting. All gradient boosted models mostly rely on the mean of EXT_SOURCE ratings.

7. **Deployment**: The successful deployment of the best-performing XGBoost model using FastAPI and Docker to Google Cloud Platform demonstrated the project's practical applicability, enabling scalable production use.


#### Suggestions for improvement:
There are still a lot that could be experimented with and improved in this project. Below are some suggestions for further improvement:

* Experiment with handling skewness of numerical features using different transformations instead of winsorization of only selected features.

* Try non-tree based models like SVM, KNN, Logistic Regression with feature interactions.

* Examine if oversampling and undersampling could improve predictive performance of the model.

* Experiment with different feature encoding techniques for categorical features.

* Try removing more multicollinear features based on correlation analysis to reduce feature redundancy.

* Experiment with different missing values, outliers and anomaly detection and handling techniques, especially in the helper tables. Also applying different aggregations to helper tables.

* Explore additional feature engineering techniques, including polynomial features, clustering-based features, or domain-specific transformations.