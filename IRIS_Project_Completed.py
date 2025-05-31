# IRIS Classification Project

# === Import Libraries ===
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# === Load Dataset ===
data_iris = pd.read_csv("Iris.csv")

# === Data Overview ===
print(data_iris.head())
print(data_iris.info())
print(data_iris.isnull().sum())

# === Visualizations ===
sns.pairplot(data_iris, hue='Species')
plt.show()

sns.heatmap(data_iris.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# === Encode Target Variable ===
le = LabelEncoder()
data_iris['Species'] = le.fit_transform(data_iris['Species'])

# === Feature and Target Separation ===
X = data_iris.drop(['Species'], axis=1)
y = data_iris['Species']

# === Split Data ===
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize Models ===
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier()
}

# === Train, Predict, Evaluate ===
results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    results.append({"Model": name, "Accuracy": acc, "R2 Score": r2, "MSE": mse})
    print(f"--- {name} ---")
    print(f"Accuracy: {acc}")
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}\n")

# === Results Summary ===
results_df = pd.DataFrame(results)
print(results_df)

# === Visualize Comparison ===
plt.figure(figsize=(10,6))
sns.barplot(data=results_df.melt(id_vars='Model', var_name='Metric', value_name='Value'),
            x='Model', y='Value', hue='Metric')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
