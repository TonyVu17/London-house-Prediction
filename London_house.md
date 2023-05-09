```python
#Import Libraries
import numpy as np 
import pandas as pd
```


```python
#Loading data file 
ld_house=pd.read_csv('D:/tableau/london house/london_house.csv')
```


```python
#Showing data
ld_house.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Property Name</th>
      <th>Price</th>
      <th>House Type</th>
      <th>Area in sq ft</th>
      <th>No. of Bedrooms</th>
      <th>No. of Bathrooms</th>
      <th>No. of Receptions</th>
      <th>Location</th>
      <th>City/County</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Queens Road</td>
      <td>1675000</td>
      <td>House</td>
      <td>2716</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>Wimbledon</td>
      <td>London</td>
      <td>SW19 8NY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Seward Street</td>
      <td>650000</td>
      <td>Flat / Apartment</td>
      <td>814</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Clerkenwell</td>
      <td>London</td>
      <td>EC1V 3PA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Hotham Road</td>
      <td>735000</td>
      <td>Flat / Apartment</td>
      <td>761</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1QL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Festing Road</td>
      <td>1765000</td>
      <td>House</td>
      <td>1986</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1LP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Spencer Walk</td>
      <td>675000</td>
      <td>Flat / Apartment</td>
      <td>700</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1PL</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Checking data
#There are missing values in Location 
ld_house.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3480 entries, 0 to 3479
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   Unnamed: 0         3480 non-null   int64 
     1   Property Name      3480 non-null   object
     2   Price              3480 non-null   int64 
     3   House Type         3480 non-null   object
     4   Area in sq ft      3480 non-null   int64 
     5   No. of Bedrooms    3480 non-null   int64 
     6   No. of Bathrooms   3480 non-null   int64 
     7   No. of Receptions  3480 non-null   int64 
     8   Location           2518 non-null   object
     9   City/County        3480 non-null   object
     10  Postal Code        3480 non-null   object
    dtypes: int64(6), object(5)
    memory usage: 299.2+ KB
    


```python
#Slit postal code to area code
ld_house['Area code']=ld_house['Postal Code'].str.split(expand=True)[0]
```


```python
#Checking data
ld_house.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Property Name</th>
      <th>Price</th>
      <th>House Type</th>
      <th>Area in sq ft</th>
      <th>No. of Bedrooms</th>
      <th>No. of Bathrooms</th>
      <th>No. of Receptions</th>
      <th>Location</th>
      <th>City/County</th>
      <th>Postal Code</th>
      <th>Area code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Queens Road</td>
      <td>1675000</td>
      <td>House</td>
      <td>2716</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>Wimbledon</td>
      <td>London</td>
      <td>SW19 8NY</td>
      <td>SW19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Seward Street</td>
      <td>650000</td>
      <td>Flat / Apartment</td>
      <td>814</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Clerkenwell</td>
      <td>London</td>
      <td>EC1V 3PA</td>
      <td>EC1V</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Hotham Road</td>
      <td>735000</td>
      <td>Flat / Apartment</td>
      <td>761</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1QL</td>
      <td>SW15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Festing Road</td>
      <td>1765000</td>
      <td>House</td>
      <td>1986</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1LP</td>
      <td>SW15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Spencer Walk</td>
      <td>675000</td>
      <td>Flat / Apartment</td>
      <td>700</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Putney</td>
      <td>London</td>
      <td>SW15 1PL</td>
      <td>SW15</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating data table to analyse
data=ld_house[['Price','House Type','Area in sq ft','No. of Bedrooms','No. of Bathrooms','No. of Receptions','Area code']]
```


```python
#Checking data table
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>House Type</th>
      <th>Area in sq ft</th>
      <th>No. of Bedrooms</th>
      <th>No. of Bathrooms</th>
      <th>No. of Receptions</th>
      <th>Area code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1675000</td>
      <td>House</td>
      <td>2716</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>SW19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>650000</td>
      <td>Flat / Apartment</td>
      <td>814</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>EC1V</td>
    </tr>
    <tr>
      <th>2</th>
      <td>735000</td>
      <td>Flat / Apartment</td>
      <td>761</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>SW15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1765000</td>
      <td>House</td>
      <td>1986</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>SW15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>675000</td>
      <td>Flat / Apartment</td>
      <td>700</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>SW15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3475</th>
      <td>3350000</td>
      <td>New development</td>
      <td>1410</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>SW6</td>
    </tr>
    <tr>
      <th>3476</th>
      <td>5275000</td>
      <td>Flat / Apartment</td>
      <td>1749</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>SW1A</td>
    </tr>
    <tr>
      <th>3477</th>
      <td>5995000</td>
      <td>House</td>
      <td>4435</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>NW11</td>
    </tr>
    <tr>
      <th>3478</th>
      <td>6300000</td>
      <td>New development</td>
      <td>1506</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>W1S</td>
    </tr>
    <tr>
      <th>3479</th>
      <td>8650000</td>
      <td>House</td>
      <td>5395</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>N6</td>
    </tr>
  </tbody>
</table>
<p>3480 rows × 7 columns</p>
</div>




```python
#Import Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```


```python
#Encode data series
le = LabelEncoder()
data['House Type']=le.fit_transform(data['House Type'])
data['Area code']=le.fit_transform(data['Area code'])
```

    C:\Users\tuana\AppData\Local\Temp\ipykernel_18588\1689077859.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data['House Type']=le.fit_transform(data['House Type'])
    C:\Users\tuana\AppData\Local\Temp\ipykernel_18588\1689077859.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data['Area code']=le.fit_transform(data['Area code'])
    


```python
#Final checking
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>House Type</th>
      <th>Area in sq ft</th>
      <th>No. of Bedrooms</th>
      <th>No. of Bathrooms</th>
      <th>No. of Receptions</th>
      <th>Area code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1675000</td>
      <td>3</td>
      <td>2716</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>650000</td>
      <td>2</td>
      <td>814</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>735000</td>
      <td>2</td>
      <td>761</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1765000</td>
      <td>3</td>
      <td>1986</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>675000</td>
      <td>2</td>
      <td>700</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3475</th>
      <td>3350000</td>
      <td>5</td>
      <td>1410</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>104</td>
    </tr>
    <tr>
      <th>3476</th>
      <td>5275000</td>
      <td>2</td>
      <td>1749</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3477</th>
      <td>5995000</td>
      <td>3</td>
      <td>4435</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3478</th>
      <td>6300000</td>
      <td>5</td>
      <td>1506</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>131</td>
    </tr>
    <tr>
      <th>3479</th>
      <td>8650000</td>
      <td>3</td>
      <td>5395</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
<p>3480 rows × 7 columns</p>
</div>




```python
#Set independent variable X, dependent variable y
y =data["Price"]
X =data.drop(["Price"] ,axis="columns")
```


```python
#Get X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("data is ",data.shape)
print("X is ",X.shape)
print("X_train is ",X_train.shape)
print("y_train is ",y_train.shape)
print("X_test is ",X_test.shape)
print("y_test is ",y_test.shape)
```

    data is  (3480, 7)
    X is  (3480, 6)
    X_train is  (2958, 6)
    y_train is  (2958,)
    X_test is  (522, 6)
    y_test is  (522,)
    


```python
#Import Linear regression model
#Model fitting
from sklearn.linear_model import LinearRegression
model_Li = LinearRegression()
model_Li.fit(X, y)
```




    LinearRegression()




```python
#Checking test score for Linear model
print('Linear model Train Score is : ' , model_Li.score(X_train, y_train))
print('Linear model Test Score is : ' , model_Li.score(X_test, y_test))
```

    Linear model Train Score is :  0.5057343583508348
    Linear model Test Score is :  0.43782939181673963
    


```python
#Import Logistic regresstion model
#Model fitting
from sklearn.linear_model import LogisticRegression
model_Lg = LinearRegression()
model_Lg.fit(X, y)
```




    LinearRegression()




```python
#Checking test score for Logistic model
print('Logistic model Train Score is : ' , model_Lg.score(X_train, y_train))
print('Logistic model Test Score is : ' , model_Lg.score(X_test, y_test))
```

    Logistic model Train Score is :  0.5057343583508348
    Logistic model Test Score is :  0.43782939181673963
    


```python
#Import Random Forest Regression model
#Model fitting
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
model_Rg = RandomForestRegressor(n_estimators=100, random_state=42)
model_Rg.fit(X, y)
```




    RandomForestRegressor(random_state=42)




```python
#Checking test score for Random Forest model
print('Random Forest model Train Score is : ' , model_Rg.score(X_train, y_train))
print('Random Forest Test Score is : ' , model_Rg.score(X_test, y_test))
```

    Random Forest model Train Score is :  0.9602056506940301
    Random Forest Test Score is :  0.9642246988082807
    


```python
#Predict price of London house
X_pre=[[3,500,2,2,1,90]]
```


```python
price_pre=model_Rg.predict(X_pre)
price_pre
```

    C:\Users\tuana\anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
      warnings.warn(
    




    array([471849.5])




```python
#Conclusion:
#Random Forest Regression beat other two model with 96% test scores
```
