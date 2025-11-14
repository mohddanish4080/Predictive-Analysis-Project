# ------------------------------
# HOUSE PRICE PREDICTION PROJECT
# ------------------------------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# STEP 1: Create a Sample Dataset
# --------------------------------
# House size (sq.ft) vs Price (in lakhs)

data = {
    "House_Size": [600, 750, 800, 900, 1000, 1100, 1200, 1500, 1800, 2000, 2200, 2500],
    "House_Price": [20, 28, 30, 35, 40, 45, 50, 65, 80, 90, 100, 120]
}

df = pd.DataFrame(data)
print("Sample Dataset:")
print(df)

# --------------------------------
# STEP 2: Define Features & Labels
# --------------------------------
X = df[["House_Size"]]       # Feature
y = df["House_Price"]        # Target variable

# --------------------------------
# STEP 3: Split Data (Train/Test)
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# STEP 4: Train Linear Regression
# --------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------
# STEP 5: Predict on Test Data
# --------------------------------
y_pred = model.predict(X_test)

# --------------------------------
# STEP 6: Evaluate Model
# --------------------------------
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# --------------------------------
# STEP 7: Predict For New House
# --------------------------------
new_house_size = 1350  # sq.ft

predicted_price = model.predict([[new_house_size]])
print("\nPredicted Price for", new_house_size, "sq.ft house =",
      round(predicted_price[0], 2), "lakhs")

# --------------------------------
# Visualization (Optional)
# --------------------------------
plt.scatter(df["House_Size"], df["House_Price"])
plt.plot(df["House_Size"], model.predict(df[["House_Size"]]))
plt.xlabel("House Size (sq.ft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price Prediction Model")
plt.show()
