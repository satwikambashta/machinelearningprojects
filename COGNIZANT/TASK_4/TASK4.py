# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load sales data
sales=pd.read_csv("4.2 sales.csv")
# Drop unnecessary columns
sales.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
# Display first 10 rows
sales.head(10)

# Load stock data
stock=pd.read_csv("4.3 sensor_stock_levels.csv")
# Drop unnecessary columns
stock.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
# Display first 10 rows
stock.head(10)

# Load temperature data
temp=pd.read_csv("4.4 sensor_storage_temperature.csv")
# Drop unnecessary columns
temp.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
# Display first 10 rows
temp.head(10)

# Check data types of sales data
sales.dtypes

# Check data types of stock data
stock.dtypes

# Check data types of temperature data
temp.dtypes

# Merge the three dataframes on common columns
df_merged = stock.merge(sales, on=['timestamp', 'product_id'], how='inner')
df_merged = df_merged.merge(temp, on=['timestamp', 'id'], how='inner')

# Display first 50 rows of merged data
df_merged.head(50)

# Display last 50 rows of merged data
df_merged.tail(50)




#OUR DATASET COMES OUT EMPTY SINCE THE GIVEN DATASETS ARE FAULTY. HOWVER, THE BELOW CODE SHOULD WORK NORMALLY IF THE DATASET IS CORRECT.
#DATASETS SALES, SENSOR STORAGE TEMP AND SENSOR STOCK VALUES do not have any matching values on common fields to make merging them possible.




# Compute correlation matrix
corr_matrix = df_merged.corr()
# Set diagonal values to 0 for better visualization
for x in range(corr_matrix.shape[0]):
    corr_matrix.iloc[x,x] = 0.0

# Plot correlation matrix as a heatmap
fig, ax = plt.subplots(figsize=(16, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# Compute pairwise maximal correlations
corr_max  = corr_matrix.abs().max().to_frame()
corr_id_max = corr_matrix.abs().idxmax().to_frame()

# Aggregate and process dataframe
pair_features_corr = pd.merge(corr_id_max, corr_max, on = corr_max.index)
pair_features_corr = pair_features_corr.rename(columns = {'key_0':'Feature_one', '0_x':'Feature_two', '0_y':'correlation'})                                                .sort_values('correlation', ascending=False)                                                .reset_index().drop('index', axis=1)
pair_features_corr

# Compute skewness of columns
skew_columns = (df_merged
                .skew()
                .sort_values(ascending=False)).to_frame("skewness_value")
skew_columns

# Define features
X = df_merged[['unit_price', 'quantity']]  # Features
y = df_merged['total']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import necessary libraries for Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable for Linear Regression
X = df_merged[['unit_price', 'quantity']]  # Features
y = df_merged['total']  # Target variable

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Import necessary libraries for CatBoost Regressor
from catboost import CatBoostRegressor


# Create a CatBoost Regressor model
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Import necessary libraries for Random Forest
from sklearn.ensemble import RandomForestRegressor


# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

#now we compare the values among the three models used above: Linear regression, Catboost regressor and random forest regressor.
#after training the output values of the model that comes closest to the testing values will be chosen as the model required.
#this should be done already but cannot be done since the dataset is not correct.
#it should then be converted to the required joblib file