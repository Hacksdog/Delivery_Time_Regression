import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("C:/Users/sayan/Downloads/regression assignment/delivery_time.csv")
print("First 5 rows of data:")
print(df.head())

df = df.rename(columns={"Delivery Time":"Delivery_time", "Sorting Time": "Sorting_time"})
print("\nInfo:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nCorrelation between Sorting_Time and Delivery_Time:")
print(df[["Sorting_time","Delivery_time"]].corr())

plt.scatter(df["Sorting_time"], df["Delivery_time"])
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Sorting Time vs Delivery Time")
plt.show()

x = df[["Sorting_time"]]
y=df["Delivery_time"]

model = LinearRegression()
model.fit(x,y)
m = model.coef_[0]
c = model.intercept_
y_pred = model.predict(x)
r2 = r2_score(y, y_pred)

print(f"R-squared: {r2:.4f}")

new_sorting_times = pd.DataFrame({"Sorting_time": [5, 8, 12]})
print(new_sorting_times)
delivery_predict = model.predict(new_sorting_times)
print(delivery_predict)






