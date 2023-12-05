```python
import os
os.chdir("C:\\Users\\JAYNE\\Downloads\individual+household+electric+power+consumption")
```


```python
import pandas as pd
import numpy as np
import seaborn as sns
```


```python
file_name = 'household_power_consumption.txt'
```


```python
df=pd.read_csv(file_name, delimiter=';')
```

    C:\Users\JAYNE\AppData\Local\Temp\ipykernel_13112\1307184513.py:1: DtypeWarning: Columns (2,3,4,5,6,7) have mixed types. Specify dtype option on import or set low_memory=False.
      df=pd.read_csv(file_name, delimiter=';')
    


```python
new_file_path = 'household_power_consumption.csv'
df.to_csv(new_file_path, sep=':', index=False)
```


```python
df.head()
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
      <th>Date</th>
      <th>Time</th>
      <th>Global_active_power</th>
      <th>Global_reactive_power</th>
      <th>Voltage</th>
      <th>Global_intensity</th>
      <th>Sub_metering_1</th>
      <th>Sub_metering_2</th>
      <th>Sub_metering_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16/12/2006</td>
      <td>17:24:00</td>
      <td>4.216</td>
      <td>0.418</td>
      <td>234.840</td>
      <td>18.400</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16/12/2006</td>
      <td>17:25:00</td>
      <td>5.360</td>
      <td>0.436</td>
      <td>233.630</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16/12/2006</td>
      <td>17:26:00</td>
      <td>5.374</td>
      <td>0.498</td>
      <td>233.290</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16/12/2006</td>
      <td>17:27:00</td>
      <td>5.388</td>
      <td>0.502</td>
      <td>233.740</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16/12/2006</td>
      <td>17:28:00</td>
      <td>3.666</td>
      <td>0.528</td>
      <td>235.680</td>
      <td>15.800</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().values.any() 
```




    True




```python
df.isnull().sum()
```




    Date                         0
    Time                         0
    Global_active_power          0
    Global_reactive_power        0
    Voltage                      0
    Global_intensity             0
    Sub_metering_1               0
    Sub_metering_2               0
    Sub_metering_3           25979
    dtype: int64




```python
df.describe(include='all')
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
      <th>Date</th>
      <th>Time</th>
      <th>Global_active_power</th>
      <th>Global_reactive_power</th>
      <th>Voltage</th>
      <th>Global_intensity</th>
      <th>Sub_metering_1</th>
      <th>Sub_metering_2</th>
      <th>Sub_metering_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2075259</td>
      <td>2.049280e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1442</td>
      <td>1440</td>
      <td>6534</td>
      <td>896</td>
      <td>5168</td>
      <td>377</td>
      <td>153</td>
      <td>145</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>6/12/2008</td>
      <td>17:24:00</td>
      <td>?</td>
      <td>0.000</td>
      <td>?</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1440</td>
      <td>1442</td>
      <td>25979</td>
      <td>472786</td>
      <td>25979</td>
      <td>169406</td>
      <td>1840611</td>
      <td>1408274</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.458447e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.437154e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.700000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.100000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2075259 entries, 0 to 2075258
    Data columns (total 9 columns):
     #   Column                 Dtype  
    ---  ------                 -----  
     0   Date                   object 
     1   Time                   object 
     2   Global_active_power    object 
     3   Global_reactive_power  object 
     4   Voltage                object 
     5   Global_intensity       object 
     6   Sub_metering_1         object 
     7   Sub_metering_2         object 
     8   Sub_metering_3         float64
    dtypes: float64(1), object(8)
    memory usage: 142.5+ MB
    


```python
df.rename(columns={'Time': 'dt'}, inplace=True)
df.head()
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
      <th>Date</th>
      <th>dt</th>
      <th>Global_active_power</th>
      <th>Global_reactive_power</th>
      <th>Voltage</th>
      <th>Global_intensity</th>
      <th>Sub_metering_1</th>
      <th>Sub_metering_2</th>
      <th>Sub_metering_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16/12/2006</td>
      <td>17:24:00</td>
      <td>4.216</td>
      <td>0.418</td>
      <td>234.840</td>
      <td>18.400</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16/12/2006</td>
      <td>17:25:00</td>
      <td>5.360</td>
      <td>0.436</td>
      <td>233.630</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16/12/2006</td>
      <td>17:26:00</td>
      <td>5.374</td>
      <td>0.498</td>
      <td>233.290</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16/12/2006</td>
      <td>17:27:00</td>
      <td>5.388</td>
      <td>0.502</td>
      <td>233.740</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16/12/2006</td>
      <td>17:28:00</td>
      <td>3.666</td>
      <td>0.528</td>
      <td>235.680</td>
      <td>15.800</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = df.fillna(method= 'ffill' )
df1.isnull().sum()
```




    Date                     0
    dt                       0
    Global_active_power      0
    Global_reactive_power    0
    Voltage                  0
    Global_intensity         0
    Sub_metering_1           0
    Sub_metering_2           0
    Sub_metering_3           0
    dtype: int64




```python
df1.head()
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
      <th>Date</th>
      <th>dt</th>
      <th>Global_active_power</th>
      <th>Global_reactive_power</th>
      <th>Voltage</th>
      <th>Global_intensity</th>
      <th>Sub_metering_1</th>
      <th>Sub_metering_2</th>
      <th>Sub_metering_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16/12/2006</td>
      <td>17:24:00</td>
      <td>4.216</td>
      <td>0.418</td>
      <td>234.840</td>
      <td>18.400</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16/12/2006</td>
      <td>17:25:00</td>
      <td>5.360</td>
      <td>0.436</td>
      <td>233.630</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16/12/2006</td>
      <td>17:26:00</td>
      <td>5.374</td>
      <td>0.498</td>
      <td>233.290</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>2.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16/12/2006</td>
      <td>17:27:00</td>
      <td>5.388</td>
      <td>0.502</td>
      <td>233.740</td>
      <td>23.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16/12/2006</td>
      <td>17:28:00</td>
      <td>3.666</td>
      <td>0.528</td>
      <td>235.680</td>
      <td>15.800</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from fbprophet import Prophet
import matplotlib.pyplot as plt

df1['dt'] = pd.to_datetime(df['dt'])
df1 = df.sort_values('dt')

length_of_dataset = len(df1)

train = df1.iloc[:length_of_dataset - 300]
test = df1.iloc[length_of_dataset - 300:]

train_prophet = train[['dt', 'Global_active_power']]
train_prophet = train_prophet.rename(columns={'dt': 'ds', 'Global_active_power': 'y'})

model = Prophet()

model.fit(train_prophet)

future = model.make_future_dataframe(periods=300, freq='D')  

forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(12, 6))
model.plot(forecast, ax=ax)
ax.plot(test['dt'], test['global_active_power'], label='Actual', color='red')
ax.legend()
plt.show()

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 from fbprophet import Prophet
          2 import matplotlib.pyplot as plt
          4 df['dt'] = pd.to_datetime(df['dt'])
    

    ModuleNotFoundError: No module named 'fbprophet'



```python
train = df1.iloc[:-300]
test = df1.iloc[-300:]

train_prophet = train[['dt', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
train_prophet = train_prophet.rename(columns={'dt': 'ds', 'Global_active_power': 'y', 'Global_reactive_power': 'add1', 'Voltage': 'add2', 'Global_intensity': 'add3', 'Sub_metering_1': 'add4', 'Sub_metering_2': 'add5', 'Sub_metering_3': 'add6'})

model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model.add_regressor('add1')
model.add_regressor('add2')
model.add_regressor('add3')
model.add_regressor('add4')
model.add_regressor('add5')
model.add_regressor('add6')

model.fit(train_prophet)

future = model.make_future_dataframe(periods=300, freq='D')
future[['add1', 'add2', 'add3', 'add4', 'add5', 'add6']] = test[['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].values

forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(12, 6))
model.plot(forecast, ax=ax)
ax.plot(test['dt'], test['global_active_power'], label='Actual', color='red')
ax.legend()
plt.show()

```
