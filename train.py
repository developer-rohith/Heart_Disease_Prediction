import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('heart.csv')
print(df.head())

df=df.drop("Unnamed: 0",axis=1)
print(df.head())
#sns.lmplot(x='biking', y='heart.disease',data=df)
#sns.lmplot(x='smoking', y='heart.disease',data=df)
#plt.show()

x_df=df.drop("heart.disease",axis=1)
y_df=df['heart.disease']


X_train, X_test, y_train, y_test = train_test_split(x_df,y_df, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

import pickle
model=pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[10.1,89.3]]))

