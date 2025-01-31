import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read combined_df from final_results.csv
combined_df = pd.read_csv('final_results.csv')

# Display the first few rows of the DataFrame
print(combined_df.head())

# Drop unnecessary columns
updated_df = combined_df.drop(columns=['name', 'subject_id', 'condition'])

# Encode the categorical labels (Big Five traits) into numerical values
label_encoders = {}
for column in ['extraversion', 'agreeableness', 'openness', 'conscientiousness', 'neuroticism']:
    le = LabelEncoder()
    updated_df[column] = le.fit_transform(updated_df[column])
    label_encoders[column] = le

# Separate features and labels
X = updated_df.drop(columns=['extraversion', 'agreeableness', 'openness', 'conscientiousness', 'neuroticism'])
y = updated_df[['extraversion', 'agreeableness', 'openness', 'conscientiousness', 'neuroticism']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier using RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy for each trait
for i, trait in enumerate(['extraversion', 'agreeableness', 'openness', 'conscientiousness', 'neuroticism']):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f'Accuracy for {trait}: {accuracy:.2f}')

# Overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {overall_accuracy:.2f}')

# Plot confusion matrices for each trait
for i, trait in enumerate(['extraversion', 'agreeableness', 'openness', 'conscientiousness', 'neuroticism']):
    cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders[trait].classes_, yticklabels=label_encoders[trait].classes_)
    plt.title(f'Confusion Matrix for {trait}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()