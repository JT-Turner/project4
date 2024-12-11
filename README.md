# Project4: Diabetes Prediction Using Machine Learning

## Project Proposal

### **Objective**
To predict the probability of diabetes based on historical data using a machine learning model.

---

### **Project Background**
The primary goal is to build a model that predicts the likelihood of developing diabetes based on various features (such as age, BMI, blood sugar levels, etc.). Below are the key questions our team will explore using machine learning techniques learned in class:

- **How well does the BRFSS survey provide accurate predictions of diabetes?** (Sara)  
- **What risk factors are most predictive of diabetes risk?** (Sanmi)  
- **Can a subset of risk factors accurately predict diabetes?** (JT)  
- **How does socioeconomic status affect access to healthy food and healthcare?** (Marco)  
- **What features are most indicative of someone getting diabetes?** (Zaineb)  
- **How do mental health and insulin resistance strongly influence diabetes risk?** (Shriya)  

---

### **Machine Learning Techniques**
The team will utilize one of the following supervised or unsupervised learning techniques:

#### **Supervised Learning**  
1. **Logistic Regression:** Binary classification predicting diabetes status (0 = No, 1 = Yes).  
   `from sklearn.linear_model import LogisticRegression`  
2. **Decision Trees:** Simple, interpretable models based on decision rules.  
3. **Random Forest:** Ensemble method combining multiple decision trees to improve accuracy and reduce overfitting.  
4. **Gradient Boosting Machines (GBM):** Techniques like XGBoost, LightGBM, or CatBoost to handle complex relationships.  
5. **Support Vector Machines (SVM):** Classification using hyperplanes for separating classes.  
6. **Neural Networks:** Deep learning models for large datasets and complex relationships.  
7. **K-Nearest Neighbors (KNN):** Classifies based on data point similarity.  

#### **Unsupervised Learning**  
- **Clustering (e.g., K-Means):** For segmenting populations based on diabetes risk factors.

---

### **Evaluation Metrics**
To measure model performance, we will evaluate:
- **Accuracy, Precision, Recall, and F1-Score**
- **ROC-AUC:** Measures the model's ability to distinguish between diabetic and non-diabetic individuals.

---

## Data and Data Delivery

### **Data Sources**
1. **Public Health Datasets:**  
   [Diabetes Health Indicators Dataset (BRFSS)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv)  
2. **Food Environment Atlas:**  
   [Data on Access to Healthy Foods](https://www.ers.usda.gov/data-products/food-environment-atlas/data-access-and-documentation-downloads/#Current%20Version)  

### **Data Preprocessing Steps**
1. **Data Cleaning:** Handle missing values, outliers, and duplicates.  
2. **Feature Engineering:** Create derived variables (e.g., BMI categories, age groups).  
3. **Normalization/Standardization:** Scale numerical features to a uniform range.  
4. **Categorical Encoding:** Convert categorical variables into numeric formats (e.g., one-hot encoding).  

### **Data Delivery**
- **Real-Time APIs:** Deliver model predictions via RESTful APIs for real-time inputs.  
- **Batch Processing:** Process data periodically (e.g., daily or weekly).  
- **Cloud Storage:** Store large datasets securely on platforms like AWS, Google Cloud, or Azure.

---

## Back-End (ETL Process)

### **ETL Workflow**
1. **Extract:**  
   - Pull data from CSV files using Python libraries (`pandas`, `SQLAlchemy`, etc.).  

2. **Transform:**  
   - Address missing values, outliers, and inconsistencies.  
   - Engineer features (e.g., binning continuous variables, encoding categories).  
   - Normalize numerical variables for scaling.

3. **Load:**  
   - Store processed data in databases like PostgreSQL or MongoDB.

### **Backend Frameworks**
- **Flask/FastAPI:** Serve the ML model via API endpoints for real-time predictions.  
- **Django:** For integrated web applications combining backend and frontend.

---

## Visualizations

### **Risk Prediction Visuals**
1. **ROC Curve:** Evaluate the model's ability to distinguish between classes.  
2. **Confusion Matrix:** Display true positives, true negatives, false positives, and false negatives.  

### **Feature Importance**
1. **Bar Graphs:** Highlight key features like BMI, glucose levels, etc.  
2. **SHAP Values:** Interpret individual predictions and visualize feature impact.

### **Exploratory Data Analysis (EDA)**
- **Histograms/Boxplots:** Visualize distributions of glucose levels, BMI, etc.  
- **Pair Plots:** Explore relationships between features (e.g., glucose vs. BMI vs. age).

### **Dashboarding Options**
- **Tableau:** Interactive dashboards for filtering and exploring results.  
- **Plotly/Dash:** Web-based dashboards for trends and predictions.  

### **Web-Based Visualization**
- **Heatmaps:** Show correlations between diabetes risk factors (e.g., glucose vs. BMI).

--- 

Let us know if you have feedback or suggestions for improving the project!
