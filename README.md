# project4
project4

Project Proposal Objective: To predict the probability of diabetes based on a machine model using historical data
 Project Background: The primary goal is to build a model that can predict the likelihood of a person developing diabetes based on various features (such as age, BMI, blood sugar levels, etc.). Below are the keys questions that the team will explore using machine learning techniques learned from our class:
How well does this survey from the BRFSS provide accurate predictions of whether an individual has diabetes? (Sara)
What risk factors are most predictive of diabetes risk? (Leader)
Can we use a subset of the risk factors to accurately predict whether an individual has diabetes? (JT)
How does socioeconomic status affect access to healthy food and healthcare? (Marco)
What features are most indicative of someone getting diabetes? (Zaineb)
How does mental health and insulin resistance strongly influence diabetes risk? (Shriya)
One of the following Supervised Learning Techniques will be used:
Logistic Regression: Use binary classification predicting whether a person has diabetes (1) or not (0). (i.e. from sklearn.linear_model import LogisticRegression)
Decision Trees: Simple, interpretable models that can help make predictions based on hierarchical decision rules.
Random Forest: An ensemble method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
Gradient Boosting Machines (GBM): Such as XGBoost, LightGBM, or CatBoost, which can handle complex relationships in the data and often perform well in prediction tasks.
Support Vector Machines (SVM): Used for classification by finding a hyperplane that best separates the classes.
Neural Networks: Deep learning models can be used if there is a large amount of data and complex relationships, although they require significant computational resources.
K-Nearest Neighbors (KNN): A simple method that classifies based on the similarity between data points.
Unsupervised Learning:
Clustering (e.g., K-Means): If you don't have labeled data, clustering can help identify patterns in the data. This could be useful for segmenting populations into groups based on diabetes risk factors.
Evaluation Metrics:
Accuracy, Precision, Recall, F1-Score: Important for evaluating classification models.
ROC-AUC: Measures how well the model can distinguish between the classes (diabetic vs. non-diabetic).

2. Data and Data Delivery
For a diabetes prediction model, the data typically involves health-related features such as age, BMI, blood pressure, glucose levels, and family history. Key considerations include:
Data Sources:
Public Health Datasets: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv
Data Collection and Preprocessing:
Data Cleaning: Handle missing values (imputation), outliers, and duplicates.
Feature Engineering: Create new features from existing ones (e.g., age groups, BMI categories).
Normalization/Standardization: Scale numerical features to a similar range (e.g., Min-Max scaling or Standard scaling).
Categorical Encoding: Convert categorical variables (like gender, smoking status) into numerical format (e.g., one-hot encoding).
Data Delivery:
Real-time APIs: If real-time predictions are needed, consider delivering data through RESTful APIs that expose the model predictions based on live input data.
Batch Processing: For periodic predictions, the data can be processed in batches (e.g., daily or weekly predictions) and stored in a database or served through a report.
Cloud Data Storage: Use cloud platforms (AWS, Google Cloud, Azure) to store large datasets securely and ensure scalability.

3. Back-End (ETL)

The team will use the following ETL Process for Diabetes Prediction:
Extract:
We will be using CSV files to pull relevant data.
Use Python libraries (e.g., pandas, requests, SQLAlchemy)
Transform:
Data Cleaning: Address missing values, outliers, and inconsistencies.
Feature Engineering: Apply transformations to create meaningful features (e.g., categorical to numeric encoding, binning continuous variables).
Data Normalization: Standardize numerical features.
Load:
Store the processed data in a database (e.g., PostgreSQL, MongoDB).
Backend Frameworks:
Flask/FastAPI (Python): For serving the machine learning model as an API endpoint for real-time predictions.
Django: For a more comprehensive web application with integrated backend and frontend.

4. Visualizations
Team will select one of the following to present and interpret the results:
Risk Prediction Visuals:
ROC Curve: To evaluate the model's ability to distinguish between classes.
Confusion Matrix: To show the true positives, true negatives, false positives, and false negatives of the model.
Feature Importance:
Bar Graphs: To show the importance of each feature (e.g., BMI, glucose levels) in the prediction.
SHAP (Shapley Additive Explanations): A powerful method to interpret model predictions and visualize how different features impact individual predictions.
Exploratory Data Analysis (EDA):
Histograms and Boxplots: To visualize distributions of key features like glucose levels, BMI, etc.
Pair Plots: To identify relationships between multiple features (e.g., glucose vs. BMI vs. age).
Dashboarding:
Tableau: For creating interactive dashboards that allow users to filter and explore diabetes prediction results and associated risk factors.
Plotly/Dash (Python): For web-based dashboards that can visualize predictions, risk factors, and trends.
Web-based Visualization:
Heatmaps: To visualize how risk factors correlate with the likelihood of diabetes (e.g., heatmap of glucose levels vs. BMI vs. age).



