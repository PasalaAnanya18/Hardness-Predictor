import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df=pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\Datasets\cleaned_data.xlsx",engine='openpyxl')

df=df.drop(['DOI','References'],axis=1)
x = df.drop(["Hardness (HV)", "Unnamed: 0"], axis=1)
y = df["Hardness (HV)"]
x['Cu'] = pd.to_numeric(x['Cu'], errors='coerce')
x = x.dropna(subset=['Cu'])
y = y[x.index] # ensure y matches the filtered x

x['Aging'] = x['Aging'].map({'Y': 1, 'N': 0})
x['Secondar. thermo-mecha0ical process']=x['Secondar. thermo-mecha0ical process'].map({'Y': 1, 'N': 0})
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

from xgboost import XGBRegressor

model1 = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model1.fit(xtrain,ytrain)
mae=mean_absolute_error(ytest,model1.predict(xtest))
mse=mean_squared_error(ytest,model1.predict(xtest))
rmse=np.sqrt(mse)
r2=r2_score(ytest,model1.predict(xtest))

plt.scatter(ytest,model1.predict(xtest))
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--') 
plt.xlabel('Actual Hardness')
plt.ylabel('Predicted Hardness')
plt.title('Actual vs. Predicted Values')
plt.show()

feature_importances=pd.Series(model1.feature_importances_,index=xtrain.columns)
feature_importances_sorted=feature_importances.sort_values(ascending=False)
feature_importances_sorted.plot(kind='bar')
plt.ylabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import joblib
joblib.dump(model1, 'xgboost_hardness_model.pkl')