import pandas as pd

# Load dataset
df = pd.read_csv("heart.csv")

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Preview first few rows
print(df.head())


from sklearn.model_selection import train_test_split

# Features and target
X = df.drop('condition', axis=1)
y = df['condition']

# Stratified train-test split (to keep class distribution balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from sklearn.preprocessing import StandardScaler

# Scale features for better model performance
scaler = StandardScaler()

# Fit on train data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with balanced class weights
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear',
    random_state=42
)

# Train the model
model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

