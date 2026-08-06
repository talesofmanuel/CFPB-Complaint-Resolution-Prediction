# CFPB-Complaint-Resolution-Prediction
MSc Data Science Dissertation - Predicting Consumer Complaint Resolution Outcomes Using Fused NLP-Derived Sentiment and Structured Metadata
CFPB Complaint Resolution Prediction 

MSc Data Science Dissertation
Coventry University, 2025/2026

Student: Onyekachukwu Emmanuel Ogbodo

Supervisor: Dr. Stenford Ruvinga

Project Title: Predicting Consumer Complaint Resolution Outcomes Using Fused NLP-Derived Sentiment and Structured Metadata: A Machine Learning Study on CFPB Credit Bureau Complaints.

Repository Contents:
model_results.csv - AUC-ROC, F1, precision, and recall for all nine model configurations under stratified 5-fold cross-validation.
shap_feature_importance.csv - Mean absolute SHAP values for all 17 features ranked by predictive contribution.
cfpb_vader_scores.csv - VADER compound, positive, negative, and neutral scores for all 197,895 eligible narratives.
cfpb_finbert_scores.csv - FinBERT label, confidence, and class probability scores for all 197,895 eligible narratives.

7005SCN_Dissertation.ipynb - Complete implementation notebook covering data loading, EDA, preprocessing, sentiment extraction, model training, evaluation, and SHAP analysis.

Dataset: The CFPB Consumer Complaint Database is publicly available at https://www.consumerfinance.gov/data-research/consumer-complaints/

Ethics Approval: This study received Low Risk Ethics Approval from Coventry University. Reference number P194996. Date of approval 11 June 2026.
