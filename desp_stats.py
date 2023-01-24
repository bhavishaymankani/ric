import pandas as pd

d= {'age': pd.Series([25,26,51,46]), 'rating': pd.Series([4.23,3.98,4.10,3.65])}

df = pd.DataFrame(d)

print(df)
print("sum")
print(df.sum())
print("mean")
print(df.mean())
print("std")
print(df.std())
print("describe")
print(df.describe())