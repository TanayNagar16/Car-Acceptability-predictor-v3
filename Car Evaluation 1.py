import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
print(f"Current working directory: {os.getcwd()}")

# Load and preprocess data
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car.data', names=column_names)

# Encode categorical features
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Exploratory Data Analysis (Basic)
sns.countplot(x='class', data=df)
plt.title('Distribution of Car Acceptability Classes (Encoded)')
plt.xlabel('Class (Encoded)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='safety', y='class', data=df)
plt.title('Car Acceptability vs. Safety')
plt.xlabel('Safety (Encoded)')
plt.ylabel('Class (Encoded)')
plt.show()

# Create and train the model
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)

# Evaluate the model (optional here)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoders['class'].classes_,
            yticklabels=encoders['class'].classes_)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

# Export the model and the encoders
filename_model = 'car_evaluation_model.pkl'
pickle.dump(model, open(filename_model, 'wb'))
print(f'\nModel saved as {filename_model}')

filename_encoders = 'car_evaluation_encoders.pkl'
pickle.dump(encoders, open(filename_encoders, 'wb'))
print(f'Encoders saved as {filename_encoders}')
