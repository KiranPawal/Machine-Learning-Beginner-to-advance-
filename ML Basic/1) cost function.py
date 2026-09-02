#i gave try to find cost function by using this car price prediction model 
#first is i gave trin the car price prediction using the regressioon method and then try to find the cost function using the mean square error method

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler

df=pd.read_csv('D:\ML Course\Datasets\car_price_prediction.csv')
print(df.head())

print(df.isnull().sum().sum())

print(df.columns)
inputs=df[['Manufacturer','Prod. year','Category','Color','Airbags','Mileage']]
target=df['Price']
print(inputs.head())

inputs['Prod. year']=2026-inputs['Prod. year']

encoder = LabelEncoder()
inputs['Manufacturer'] = encoder.fit_transform(inputs['Manufacturer'])
inputs['Category']=encoder.fit_transform(inputs['Category'])
inputs['Color']=encoder.fit_transform(inputs['Color'])
print(inputs.head())

scaler=StandardScaler()
inputs['Prod. year']=scaler.fit_transform(inputs[['Prod. year']])
inputs['Airbags']=scaler.fit_transform(inputs[['Airbags']])

inputs['Mileage'] = inputs['Mileage'].str.replace(' km', '', regex=False)
inputs['Mileage'] = pd.to_numeric(inputs['Mileage'])
inputs['Mileage']=scaler.fit_transform(inputs[['Mileage']])
print(inputs.head())


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=40)
print(x_train)
print(y_test)


model = XGBRegressor(
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
model.fit(x_train,y_train)
print(model.predict(x_test))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(x_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("-----------------")
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)








# ============================================================
# NEW CAR PRICE PREDICTION
# Add this block BEFORE your existing prediction section
# ============================================================

print("\n======================================")
print("       CAR PRICE PREDICTION")
print("======================================")

manufacturer_encoder = LabelEncoder()
category_encoder = LabelEncoder()
color_encoder = LabelEncoder()

manufacturer_encoder.fit(df['Manufacturer'])
category_encoder.fit(df['Category'])
color_encoder.fit(df['Color'])


year_scaler = StandardScaler()
airbag_scaler = StandardScaler()
mileage_scaler = StandardScaler()

# Car age
car_age_data = 2026 - df['Prod. year']
year_scaler.fit(car_age_data.to_numpy().reshape(-1, 1))

# Airbags
airbag_scaler.fit(
    df[['Airbags']]
)

# Mileage
mileage_data = (
    df['Mileage']
    .str.replace(' km', '', regex=False)
    .astype(float)
)

mileage_scaler.fit(
    mileage_data.to_numpy().reshape(-1, 1)
)


# ------------------------------------------------------------
# Take ORIGINAL input from user
# ------------------------------------------------------------

manufacturer = input("Enter Manufacturer: ").strip()
prod_year = int(input("Enter Production Year: "))
category = input("Enter Category: ").strip()
color = input("Enter Color: ").strip()
airbags = float(input("Enter Number of Airbags: "))
mileage = float(input("Enter Mileage (km): "))


# ------------------------------------------------------------
# Find exact dataset value
# This allows Honda / HONDA / honda
# ------------------------------------------------------------

def find_value(user_input, available_values):
    for value in available_values:
        if str(value).strip().lower() == user_input.lower():
            return value
    return None


manufacturer = find_value(
    manufacturer,
    manufacturer_encoder.classes_
)

category = find_value(
    category,
    category_encoder.classes_
)

color = find_value(
    color,
    color_encoder.classes_
)


# ------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------

if manufacturer is None:

    print("\nManufacturer not found.")

    print("Available Manufacturers:")
    print(list(manufacturer_encoder.classes_))

    exit()


if category is None:

    print("\nCategory not found.")

    print("Available Categories:")
    print(list(category_encoder.classes_))

    exit()


if color is None:

    print("\nColor not found.")

    print("Available Colors:")
    print(list(color_encoder.classes_))

    exit()


# ------------------------------------------------------------
# Encode categorical values
# ------------------------------------------------------------

manufacturer_encoded = manufacturer_encoder.transform(
    [manufacturer]
)[0]

category_encoded = category_encoder.transform(
    [category]
)[0]

color_encoded = color_encoder.transform(
    [color]
)[0]


car_age = 2026 - prod_year


prod_year_scaled = year_scaler.transform(
    [[car_age]]
)[0][0]

airbags_scaled = airbag_scaler.transform(
    [[airbags]]
)[0][0]

mileage_scaled = mileage_scaler.transform(
    [[mileage]]
)[0][0]


new_car = pd.DataFrame({
    'Manufacturer': [manufacturer_encoded],
    'Prod. year': [prod_year_scaled],
    'Category': [category_encoded],
    'Color': [color_encoded],
    'Airbags': [airbags_scaled],
    'Mileage': [mileage_scaled]
})


print("\nProcessed Input:")
print(new_car)


predicted_price = model.predict(new_car)[0]


print("\n======================================")
print("        PREDICTED CAR PRICE")
print("======================================")

print(f"Manufacturer    : {manufacturer}")
print(f"Production Year : {prod_year}")
print(f"Category        : {category}")
print(f"Color           : {color}")
print(f"Airbags         : {airbags}")
print(f"Mileage         : {mileage} km")

print("--------------------------------------")

print(f"Predicted Price : {predicted_price:,.2f}")

print("======================================")

# Stop here so your old prediction code below
# does not execute
exit()

