# ============================================================
# CAR PRICE PREDICTION
# XGBoost + Multiple Linear Regression using Gradient Descent
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    r'D:\ML Course\Datasets\car_price_prediction.csv'
)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    'Manufacturer',
    'Prod. year',
    'Category',
    'Color',
    'Airbags',
    'Mileage'
]

target_column = 'Price'

X = df[features].copy()
y = df[target_column].copy()


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Convert production year into car age
X['Prod. year'] = 2026 - X['Prod. year']

# Convert mileage from string to number
X['Mileage'] = (
    X['Mileage']
    .str.replace(' km', '', regex=False)
    .astype(float)
)


# Remove rows containing missing values
data = pd.concat([X, y], axis=1).dropna()

X = data[features].copy()
y = data[target_column].copy()


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=40
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. CATEGORICAL + NUMERICAL FEATURES
# ============================================================

categorical_features = [
    'Manufacturer',
    'Category',
    'Color'
]

numerical_features = [
    'Prod. year',
    'Airbags',
    'Mileage'
]


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ),
            categorical_features
        ),
        (
            'numerical',
            'passthrough',
            numerical_features
        )
    ]
)


# Fit ONLY on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Transform test data
X_test_encoded = preprocessor.transform(X_test)


print("\nEncoded training shape:")
print(X_train_encoded.shape)

print("Encoded testing shape:")
print(X_test_encoded.shape)


# ============================================================
# 7. FEATURE SCALING FOR GRADIENT DESCENT
# ============================================================

feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(
    X_train_encoded
)

X_test_scaled = feature_scaler.transform(
    X_test_encoded
)


# ============================================================
# 8. SCALE TARGET FOR STABLE GRADIENT DESCENT
# ============================================================

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()


# ============================================================
# 9. COST FUNCTION
# ============================================================

def compute_cost(X, y, w, b):

    m = len(X)

    # Prediction
    predictions = np.dot(X, w) + b

    # Error
    errors = predictions - y

    # Cost function
    cost = np.sum(errors ** 2) / (2 * m)

    return cost


# ============================================================
# 10. COMPUTE GRADIENT
# ============================================================

def compute_gradient(X, y, w, b):

    m = len(X)

    # Prediction
    predictions = np.dot(X, w) + b

    # Error
    errors = predictions - y

    # Gradient for weights
    dj_dw = np.dot(X.T, errors) / m

    # Gradient for bias
    dj_db = np.sum(errors) / m

    return dj_dw, dj_db


# ============================================================
# 11. GRADIENT DESCENT
# ============================================================

def gradient_descent(
    X,
    y,
    w,
    b,
    learning_rate,
    iterations
):

    cost_history = []

    for i in range(iterations):

        # Calculate gradients
        dj_dw, dj_db = compute_gradient(
            X,
            y,
            w,
            b
        )

        # Update weights
        w = w - learning_rate * dj_dw

        # Update bias
        b = b - learning_rate * dj_db

        # Calculate cost
        cost = compute_cost(
            X,
            y,
            w,
            b
        )

        cost_history.append(cost)

        # Print cost
        if i % 100 == 0:
            print(
                f"Iteration {i:4d} | "
                f"Cost = {cost:.6f}"
            )

    return w, b, cost_history


# ============================================================
# 12. INITIALIZE PARAMETERS
# ============================================================

number_of_features = X_train_scaled.shape[1]

w_initial = np.zeros(
    number_of_features
)

b_initial = 0.0

learning_rate = 0.03
iterations = 5000


# ============================================================
# 13. TRAIN USING GRADIENT DESCENT
# ============================================================

print("\n======================================")
print(" GRADIENT DESCENT TRAINING")
print("======================================")

w_final, b_final, cost_history = gradient_descent(
    X_train_scaled,
    y_train_scaled,
    w_initial,
    b_initial,
    learning_rate,
    iterations
)


# ============================================================
# 14. FINAL PARAMETERS
# ============================================================
print("\nFinal Gradient Descent Parameters")
print("----------------------------------")
print("Number of weights:", len(w_final))
print("Bias:", b_final)
print("Final Cost:", cost_history[-1])


# ============================================================
# 15. GRADIENT DESCENT PREDICTION
# ============================================================

y_train_pred_scaled = (
    np.dot(X_train_scaled, w_final)
    + b_final
)

y_test_pred_scaled = (
    np.dot(X_test_scaled, w_final)
    + b_final
)


# Convert predictions back to original price scale
y_train_pred_gd = target_scaler.inverse_transform(
    y_train_pred_scaled.reshape(-1, 1)
).ravel()

y_test_pred_gd = target_scaler.inverse_transform(
    y_test_pred_scaled.reshape(-1, 1)
).ravel()


# ============================================================
# 16. GRADIENT DESCENT EVALUATION
# ============================================================
mae_gd = mean_absolute_error(
    y_test,
    y_test_pred_gd
)

mse_gd = mean_squared_error(
    y_test,
    y_test_pred_gd
)

rmse_gd = np.sqrt(mse_gd)

r2_gd = r2_score(
    y_test,
    y_test_pred_gd
)


print("\n======================================")
print(" GRADIENT DESCENT PERFORMANCE")
print("======================================")

print("MAE  :", mae_gd)
print("MSE  :", mse_gd)
print("RMSE :", rmse_gd)
print("R2   :", r2_gd)


# ============================================================
# 17. XGBOOST MODEL
# ============================================================

print("\n======================================")
print(" XGBOOST TRAINING")
print("======================================")

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=42
)

# XGBoost does not require feature scaling
xgb_model.fit(
    X_train_encoded,
    y_train
)


# ============================================================
# 18. XGBOOST PREDICTION
# ============================================================

y_pred_xgb = xgb_model.predict(
    X_test_encoded
)


# ============================================================
# 19. XGBOOST EVALUATION
# ============================================================

mae_xgb = mean_absolute_error(
    y_test,
    y_pred_xgb
)

mse_xgb = mean_squared_error(
    y_test,
    y_pred_xgb
)

rmse_xgb = np.sqrt(mse_xgb)

r2_xgb = r2_score(
    y_test,
    y_pred_xgb
)


print("\n======================================")
print(" XGBOOST PERFORMANCE")
print("======================================")

print("MAE  :", mae_xgb)
print("MSE  :", mse_xgb)
print("RMSE :", rmse_xgb)
print("R2   :", r2_xgb)


# ============================================================
# 20. MODEL COMPARISON
# ============================================================

print("\n======================================")
print(" MODEL COMPARISON")
print("======================================")

comparison = pd.DataFrame({
    'Model': [
        'Gradient Descent Linear Regression',
        'XGBoost'
    ],
    'MAE': [
        mae_gd,
        mae_xgb
    ],
    'MSE': [
        mse_gd,
        mse_xgb
    ],
    'RMSE': [
        rmse_gd,
        rmse_xgb
    ],
    'R2': [
        r2_gd,
        r2_xgb
    ]
})

print(comparison)


# ============================================================
# 21. COST REDUCTION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cost_history,
    label='Training Cost'
)

plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.title(
    'Cost Reduction Using Gradient Descent'
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 22. NEW CAR PRICE PREDICTION
# ============================================================

print("\n======================================")
print("       CAR PRICE PREDICTION")
print("======================================")


manufacturer = input(
    "Enter Manufacturer: "
).strip()

prod_year = int(
    input("Enter Production Year: ")
)

category = input(
    "Enter Category: "
).strip()

color = input(
    "Enter Color: "
).strip()

airbags = float(
    input("Enter Number of Airbags: ")
)

mileage = float(
    input("Enter Mileage (km): ")
)


# ============================================================
# 23. CREATE NEW CAR DATA
# ============================================================

new_car = pd.DataFrame({

    'Manufacturer': [
        manufacturer
    ],

    'Prod. year': [
        2026 - prod_year
    ],

    'Category': [
        category
    ],

    'Color': [
        color
    ],

    'Airbags': [
        airbags
    ],

    'Mileage': [
        mileage
    ]
})


print("\nOriginal Input:")
print(new_car)


# ============================================================
# 24. PREPROCESS NEW CAR
# ============================================================

# Use previously fitted preprocessor
new_car_encoded = preprocessor.transform(
    new_car
)


# ============================================================
# 25. GRADIENT DESCENT PREDICTION
# ============================================================

new_car_scaled = feature_scaler.transform(
    new_car_encoded
)

new_car_pred_scaled = (
    np.dot(
        new_car_scaled,
        w_final
    )
    + b_final
)


# Convert back to actual price
new_car_pred_scaled = (
    np.dot(new_car_scaled, w_final) + b_final
)

new_car_pred_scaled = float(new_car_pred_scaled[0])

new_car_prediction_gd = target_scaler.inverse_transform(
    np.array([[new_car_pred_scaled]])
)[0, 0]


# ============================================================
# 26. XGBOOST PREDICTION
# ============================================================

new_car_prediction_xgb = xgb_model.predict(
    new_car_encoded
)[0]


# ============================================================
# 27. FINAL RESULT
# ============================================================

print("\n======================================")
print("        PREDICTED CAR PRICE")
print("======================================")

print(
    f"Manufacturer    : {manufacturer}"
)

print(
    f"Production Year : {prod_year}"
)

print(
    f"Category        : {category}"
)

print(
    f"Color           : {color}"
)

print(
    f"Airbags         : {airbags}"
)

print(
    f"Mileage         : {mileage} km"
)

print("--------------------------------------")

print(
    f"Gradient Descent Price : "
    f"{new_car_prediction_gd:,.2f}"
)

print(
    f"XGBoost Price          : "
    f"{new_car_prediction_xgb:,.2f}"
)

print("======================================")