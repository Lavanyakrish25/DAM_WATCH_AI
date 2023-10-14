
import numpy  as np
import pandas as pd
import sklearn
import pickle

data=pd.read_csv(r"D:\Damn watch -ML\TN Dam dataset.csv")
data


data['Name'].unique()

data.isnull().sum()


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)


data.isnull().sum()

data['Status'].dropna(inplace=True)

data['Status'].isnull().sum()


data


data['Name'].unique()

data['Purpose'].unique()


data['Completion Year'].unique()

data['Length (m)'].unique()

data['Max Height above Foundation (m)'].unique()

data['Basin'].unique()

data['Status'].unique()

data['Type'].unique()

string_features = ['Name','Purpose','River','Nearest City','District','Basin','Status','Type']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in string_features:
    data[feature] = le.fit_transform(data[feature])

data.columns

for feature in string_features:
    data[feature]=data[feature].astype('float')


data_scaled = data.copy()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled= scaler.fit_transform(data_scaled)


data1=pd.DataFrame(data_scaled)
data1.columns=data.columns

data1


data1['Name'].values


data1['Purpose'].values

data1['River'].values


data1['Distance in km'].values


data1['Nearest City'].values

data1['District'].values

data1['Basin'].values

data1['Status'].values

data1['Completion Year'].values

data1['Type'].values

data1['Length (m)'].values

def calculate_risk_score(row):
    risk_score = (row['Length (m)'] * 0.5) + (row['Max Height above Foundation (m)'] * 1.5)
    return risk_score

data1['Risk Score'] = data1.apply(calculate_risk_score, axis=1)
risk_threshold = 100
high_risk_dams = data1[data1['Risk Score'] > risk_threshold]
print("High-Risk Dams:")
print(high_risk_dams[['Name', 'Risk Score']])


data1


x=data1[['Name','Purpose','Completion Year','Type','Length (m)','Max Height above Foundation (m)']]


y=data1[['Risk Score']]

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)


x_train.shape


x_test.shape


y_train.shape


y_test.shape

from sklearn.linear_model import LinearRegression


regressor=LinearRegression().fit(x_train,y_train)
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
y_pred=regressor.predict(x_test)

y_pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error,r2_score


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2) Score: {r2}")


print('Accuracy:', np.mean(y_pred== y_test))

new_data=[-1.655816,0.734358,-0.613430,0.174078,1.363867,0.503445,]
predicted_species=regressor.predict([new_data])[0]
print('Risk Score:',predicted_species)