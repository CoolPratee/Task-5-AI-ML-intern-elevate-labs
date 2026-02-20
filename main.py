# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("heart.csv")

# Show dataset info
print(df.head())
print(df.info())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 1. Decision Tree Model
# ===============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Prediction
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# ===============================
# 2. Control Overfitting (max_depth)
# ===============================
dt_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth.fit(X_train, y_train)

y_pred_depth = dt_depth.predict(X_test)
print("Decision Tree (max_depth=3) Accuracy:", accuracy_score(y_test, y_pred_depth))

# ===============================
# 3. Visualize Decision Tree
# ===============================
plt.figure(figsize=(20,10))
plot_tree(dt_depth, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ===============================
# 4. Random Forest Model
# ===============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ===============================
# 5. Feature Importance
# ===============================
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
feature_importance.plot(kind='bar', figsize=(10,6))
plt.title("Feature Importance")
plt.show()

# ===============================
# 6. Cross Validation
# ===============================
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
