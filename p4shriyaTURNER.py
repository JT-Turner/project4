# Step 1: Setup and Tools

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

data = pd.read_csv(r"C:\Users\jt4ha\Dark Sky Data Dropbox\JT Turner\Certifications\diabetes_012_health_indicators_BRFSS2015.csv")

# Step 2: Data Preparation

# Load and prepare the dataset
selected_columns = ['MentHlth', 'PhysHlth', 'BMI', 'GenHlth', 'Diabetes_012']
df = data[selected_columns]

# Handle missing values (though no missing values were indicated earlier)
df = df.dropna()

# Normalize continuous variables
scaler = MinMaxScaler()
df[['MentHlth', 'PhysHlth', 'BMI']] = scaler.fit_transform(df[['MentHlth', 'PhysHlth', 'BMI']])

# Encode the target variable
df['DiabetesRisk'] = df['Diabetes_012'].apply(lambda x: 1 if x == 2 else 0)
df = df.drop(columns=['Diabetes_012'])

# Step 3: EDA

# Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Boxplot: BMI vs DiabetesRisk
plt.figure(figsize=(6, 4))
sns.boxplot(x='DiabetesRisk', y='BMI', data=df)
plt.title("Boxplot of BMI by DiabetesRisk")
plt.show()

# Distribution of Mental Health Days by DiabetesRisk
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='MentHlth', hue='DiabetesRisk', kde=True, element="step", stat="density")
plt.title("Distribution of Mental Health Days by DiabetesRisk")
plt.show()

# Step 4: Advanced Data Analysis

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['MentHlth', 'PhysHlth', 'BMI']])

# Plot Clustering Results
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='BMI', y='MentHlth', hue='Cluster', palette='viridis')
plt.title("K-Means Clustering Results")
plt.show()

# Statistical Test: T-test between DiabetesRisk groups for BMI
non_diabetes = df[df['DiabetesRisk'] == 0]['BMI']
diabetes = df[df['DiabetesRisk'] == 1]['BMI']


# Step 5: Logistic Regression

# Split the data into training and test sets
y = df['DiabetesRisk']
X = df.drop(columns=['DiabetesRisk'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate the Logistic Regression model
y_pred = lr.predict(X_test)
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# Plot ROC Curve
y_pred_proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Export Data for Tableau
export_columns = ['MentHlth', 'PhysHlth', 'BMI', 'GenHlth', 'DiabetesRisk', 'Cluster']
df[export_columns].to_csv('processed_diabetes_data.csv', index=False)
print("Processed dataset exported for Tableau.")
print(f"Non-diabetes group size: {len(non_diabetes)}")
print(f"Diabetes group size: {len(diabetes)}")
print(f"Non-diabetes variance: {non_diabetes.var():.6f}")
print(f"Diabetes variance: {diabetes.var():.6f}")

from scipy.stats import ttest_ind
stat, p_value = ttest_ind(non_diabetes, diabetes)
print(f"T-test for BMI: t-statistic = {stat:.2f}, p-value = {p_value:.2e}")