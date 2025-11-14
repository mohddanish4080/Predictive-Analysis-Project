# ------------------------------
# STUDENT PERFORMANCE PREDICTION
# ------------------------------

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# STEP 1: Create a Sample Dataset
# --------------------------------

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 9, 10, 3, 4, 6, 7],
    "Attendance": [70, 75, 80, 85, 90, 92, 95, 60, 96, 98, 78, 82, 88, 91],
    "Internal_Marks": [10, 15, 16, 18, 19, 20, 21, 8, 22, 24, 14, 16, 18, 19],
    "Final_Exam_Score": [50, 55, 60, 65, 70, 78, 85, 40, 88, 92, 58, 62, 73, 80]
}

df = pd.DataFrame(data)
print("Sample Dataset:")
print(df)

# --------------------------------
# STEP 2: Define Input and Output
# --------------------------------

X = df[["Study_Hours", "Attendance", "Internal_Marks"]]  # Features
y = df["Final_Exam_Score"]                              # Target variable

# --------------------------------
# STEP 3: Split Data (Train/Test)
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# STEP 4: Train Model
# --------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------
# STEP 5: Predict on Test Data
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# STEP 6: Model Evaluation
# --------------------------------

print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# --------------------------------
# STEP 7: Predict for New Student
# --------------------------------

study_hours = 5
attendance = 85
internal_marks = 17

new_prediction = model.predict([[study_hours, attendance, internal_marks]])

print("\nPredicted Final Exam Score for New Student = ", round(new_prediction[0], 2))

# --------------------------------
# OPTIONAL: Visualization
# --------------------------------

plt.scatter(df["Study_Hours"], df["Final_Exam_Score"])
plt.xlabel("Study Hours")
plt.ylabel("Final Exam Score")
plt.title("Student Performance Correlation")
plt.show()
