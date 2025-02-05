---
layout: single
title:  "DACON XGBRegressor로 주식가격 예측하기"
categories: Competitions
# tag: [data science, 대회]
toc: true  #포스트 오른쪽에 목차
toc_sticky: true
author_profile : false
search: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


-------------
  이 포스트의 코드는 데이콘 주식 종료 가격 예측 경진대회에 참여하여 작성했던 코드 입니다.  [**대회 링크**](https://dacon.io/competitions/official/235857/overview/description)
  
  좋은 성적을 거두지는 못했지만, 짧은 시간동안 많은 노력을 들여 재미있게 참여했던 대회여서 블로그를 통해 코드를 공유합니다.  

-------------



```python
import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
```


```python
path = os.getcwd()
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))
stock_list
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
      <th>종목명</th>
      <th>종목코드</th>
      <th>상장시장</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>삼성전자</td>
      <td>005930</td>
      <td>KOSPI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SK하이닉스</td>
      <td>000660</td>
      <td>KOSPI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NAVER</td>
      <td>035420</td>
      <td>KOSPI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오</td>
      <td>035720</td>
      <td>KOSPI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>삼성바이오로직스</td>
      <td>207940</td>
      <td>KOSPI</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>맘스터치</td>
      <td>220630</td>
      <td>KOSDAQ</td>
    </tr>
    <tr>
      <th>366</th>
      <td>다날</td>
      <td>064260</td>
      <td>KOSDAQ</td>
    </tr>
    <tr>
      <th>367</th>
      <td>제이시스메디칼</td>
      <td>287410</td>
      <td>KOSDAQ</td>
    </tr>
    <tr>
      <th>368</th>
      <td>크리스에프앤씨</td>
      <td>110790</td>
      <td>KOSDAQ</td>
    </tr>
    <tr>
      <th>369</th>
      <td>쎄트렉아이</td>
      <td>099320</td>
      <td>KOSDAQ</td>
    </tr>
  </tbody>
</table>
<p>370 rows × 3 columns</p>
</div>



```python
start_date = '20210104'
end_date = '20211214'

Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns = ['Date'])
```


```python
Business_days['weekday'] = Business_days.Date.apply(lambda x:x.weekday())
Business_days['weeknum'] = Business_days.Date.apply(lambda x:x.strftime('%V'))
```


```python
Business_days
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
      <th>weekday</th>
      <th>weeknum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>0</td>
      <td>01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>1</td>
      <td>01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>2</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>3</td>
      <td>01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>4</td>
      <td>01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2021-12-08</td>
      <td>2</td>
      <td>49</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2021-12-09</td>
      <td>3</td>
      <td>49</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2021-12-10</td>
      <td>4</td>
      <td>49</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2021-12-13</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2021-12-14</td>
      <td>1</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
<p>247 rows × 3 columns</p>
</div>


- KOSPI 정보 추가



```python
KOSPI = fdr.DataReader('KS11', start_date, end_date).reset_index()
KOSPI.name = 'kospi'
```


```python
def col_rename(data_set):
    for i in data_set.columns:
        if i =='Date':
            pass
        else:
            data_set.rename(columns={i:data_set.name+'_'+i}, inplace=True)
    return data_set        
```


```python
col_rename(KOSPI)
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
      <th>kospi_Close</th>
      <th>kospi_Open</th>
      <th>kospi_High</th>
      <th>kospi_Low</th>
      <th>kospi_Volume</th>
      <th>kospi_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>2944.45</td>
      <td>2874.50</td>
      <td>2946.54</td>
      <td>2869.11</td>
      <td>1.030000e+09</td>
      <td>0.0247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>2990.57</td>
      <td>2943.67</td>
      <td>2990.57</td>
      <td>2921.84</td>
      <td>1.520000e+09</td>
      <td>0.0157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>2968.21</td>
      <td>2993.34</td>
      <td>3027.16</td>
      <td>2961.37</td>
      <td>1.790000e+09</td>
      <td>-0.0075</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>3031.68</td>
      <td>2980.75</td>
      <td>3055.28</td>
      <td>2980.75</td>
      <td>1.520000e+09</td>
      <td>0.0214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>3152.18</td>
      <td>3040.11</td>
      <td>3161.11</td>
      <td>3040.11</td>
      <td>1.300000e+09</td>
      <td>0.0397</td>
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
      <th>231</th>
      <td>2021-12-08</td>
      <td>3001.80</td>
      <td>3017.93</td>
      <td>3036.13</td>
      <td>2995.34</td>
      <td>4.882500e+08</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>232</th>
      <td>2021-12-09</td>
      <td>3029.57</td>
      <td>3007.00</td>
      <td>3029.57</td>
      <td>3001.55</td>
      <td>5.134400e+08</td>
      <td>0.0093</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2021-12-10</td>
      <td>3010.23</td>
      <td>3008.70</td>
      <td>3017.64</td>
      <td>2998.29</td>
      <td>4.516000e+08</td>
      <td>-0.0064</td>
    </tr>
    <tr>
      <th>234</th>
      <td>2021-12-13</td>
      <td>3001.66</td>
      <td>3019.67</td>
      <td>3043.83</td>
      <td>3000.51</td>
      <td>3.758300e+08</td>
      <td>-0.0028</td>
    </tr>
    <tr>
      <th>235</th>
      <td>2021-12-14</td>
      <td>2987.95</td>
      <td>2983.95</td>
      <td>3001.70</td>
      <td>2976.16</td>
      <td>5.815700e+08</td>
      <td>-0.0046</td>
    </tr>
  </tbody>
</table>
<p>236 rows × 7 columns</p>
</div>



```python
data = pd.merge(Business_days, KOSPI, how = 'outer')
```

- KOSDAQ 정보 추가



```python
KOSDAQ = fdr.DataReader('KQ11', start_date, end_date).reset_index()
KOSDAQ.name = 'kosdaq'
```


```python
col_rename(KOSDAQ)
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
      <th>kosdaq_Close</th>
      <th>kosdaq_Open</th>
      <th>kosdaq_High</th>
      <th>kosdaq_Low</th>
      <th>kosdaq_Volume</th>
      <th>kosdaq_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>977.62</td>
      <td>968.86</td>
      <td>977.62</td>
      <td>960.52</td>
      <td>1.700000e+09</td>
      <td>0.0095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>985.76</td>
      <td>976.43</td>
      <td>985.76</td>
      <td>965.53</td>
      <td>1.810000e+09</td>
      <td>0.0083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>981.39</td>
      <td>987.25</td>
      <td>990.88</td>
      <td>977.37</td>
      <td>1.980000e+09</td>
      <td>-0.0044</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>988.86</td>
      <td>983.28</td>
      <td>993.91</td>
      <td>982.27</td>
      <td>2.260000e+09</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>987.79</td>
      <td>990.70</td>
      <td>995.22</td>
      <td>978.12</td>
      <td>2.560000e+09</td>
      <td>-0.0011</td>
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
      <th>231</th>
      <td>2021-12-08</td>
      <td>1006.04</td>
      <td>1006.61</td>
      <td>1012.64</td>
      <td>1003.92</td>
      <td>1.070000e+09</td>
      <td>0.0094</td>
    </tr>
    <tr>
      <th>232</th>
      <td>2021-12-09</td>
      <td>1022.87</td>
      <td>1009.29</td>
      <td>1022.87</td>
      <td>1009.29</td>
      <td>1.060000e+09</td>
      <td>0.0167</td>
    </tr>
    <tr>
      <th>233</th>
      <td>2021-12-10</td>
      <td>1011.57</td>
      <td>1016.34</td>
      <td>1018.42</td>
      <td>1010.02</td>
      <td>1.150000e+09</td>
      <td>-0.0110</td>
    </tr>
    <tr>
      <th>234</th>
      <td>2021-12-13</td>
      <td>1005.96</td>
      <td>1014.27</td>
      <td>1014.90</td>
      <td>1005.96</td>
      <td>1.340000e+09</td>
      <td>-0.0055</td>
    </tr>
    <tr>
      <th>235</th>
      <td>2021-12-14</td>
      <td>1002.81</td>
      <td>1001.11</td>
      <td>1007.73</td>
      <td>996.85</td>
      <td>1.080000e+09</td>
      <td>-0.0031</td>
    </tr>
  </tbody>
</table>
<p>236 rows × 7 columns</p>
</div>



```python
data = pd.merge(data, KOSDAQ, how = 'outer')
```


```python
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
      <th>Date</th>
      <th>weekday</th>
      <th>weeknum</th>
      <th>kospi_Close</th>
      <th>kospi_Open</th>
      <th>kospi_High</th>
      <th>kospi_Low</th>
      <th>kospi_Volume</th>
      <th>kospi_Change</th>
      <th>kosdaq_Close</th>
      <th>kosdaq_Open</th>
      <th>kosdaq_High</th>
      <th>kosdaq_Low</th>
      <th>kosdaq_Volume</th>
      <th>kosdaq_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>0</td>
      <td>01</td>
      <td>2944.45</td>
      <td>2874.50</td>
      <td>2946.54</td>
      <td>2869.11</td>
      <td>1.030000e+09</td>
      <td>0.0247</td>
      <td>977.62</td>
      <td>968.86</td>
      <td>977.62</td>
      <td>960.52</td>
      <td>1.700000e+09</td>
      <td>0.0095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>1</td>
      <td>01</td>
      <td>2990.57</td>
      <td>2943.67</td>
      <td>2990.57</td>
      <td>2921.84</td>
      <td>1.520000e+09</td>
      <td>0.0157</td>
      <td>985.76</td>
      <td>976.43</td>
      <td>985.76</td>
      <td>965.53</td>
      <td>1.810000e+09</td>
      <td>0.0083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>2</td>
      <td>01</td>
      <td>2968.21</td>
      <td>2993.34</td>
      <td>3027.16</td>
      <td>2961.37</td>
      <td>1.790000e+09</td>
      <td>-0.0075</td>
      <td>981.39</td>
      <td>987.25</td>
      <td>990.88</td>
      <td>977.37</td>
      <td>1.980000e+09</td>
      <td>-0.0044</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>3</td>
      <td>01</td>
      <td>3031.68</td>
      <td>2980.75</td>
      <td>3055.28</td>
      <td>2980.75</td>
      <td>1.520000e+09</td>
      <td>0.0214</td>
      <td>988.86</td>
      <td>983.28</td>
      <td>993.91</td>
      <td>982.27</td>
      <td>2.260000e+09</td>
      <td>0.0076</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>4</td>
      <td>01</td>
      <td>3152.18</td>
      <td>3040.11</td>
      <td>3161.11</td>
      <td>3040.11</td>
      <td>1.300000e+09</td>
      <td>0.0397</td>
      <td>987.79</td>
      <td>990.70</td>
      <td>995.22</td>
      <td>978.12</td>
      <td>2.560000e+09</td>
      <td>-0.0011</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2021-12-08</td>
      <td>2</td>
      <td>49</td>
      <td>3001.80</td>
      <td>3017.93</td>
      <td>3036.13</td>
      <td>2995.34</td>
      <td>4.882500e+08</td>
      <td>0.0034</td>
      <td>1006.04</td>
      <td>1006.61</td>
      <td>1012.64</td>
      <td>1003.92</td>
      <td>1.070000e+09</td>
      <td>0.0094</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2021-12-09</td>
      <td>3</td>
      <td>49</td>
      <td>3029.57</td>
      <td>3007.00</td>
      <td>3029.57</td>
      <td>3001.55</td>
      <td>5.134400e+08</td>
      <td>0.0093</td>
      <td>1022.87</td>
      <td>1009.29</td>
      <td>1022.87</td>
      <td>1009.29</td>
      <td>1.060000e+09</td>
      <td>0.0167</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2021-12-10</td>
      <td>4</td>
      <td>49</td>
      <td>3010.23</td>
      <td>3008.70</td>
      <td>3017.64</td>
      <td>2998.29</td>
      <td>4.516000e+08</td>
      <td>-0.0064</td>
      <td>1011.57</td>
      <td>1016.34</td>
      <td>1018.42</td>
      <td>1010.02</td>
      <td>1.150000e+09</td>
      <td>-0.0110</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2021-12-13</td>
      <td>0</td>
      <td>50</td>
      <td>3001.66</td>
      <td>3019.67</td>
      <td>3043.83</td>
      <td>3000.51</td>
      <td>3.758300e+08</td>
      <td>-0.0028</td>
      <td>1005.96</td>
      <td>1014.27</td>
      <td>1014.90</td>
      <td>1005.96</td>
      <td>1.340000e+09</td>
      <td>-0.0055</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2021-12-14</td>
      <td>1</td>
      <td>50</td>
      <td>2987.95</td>
      <td>2983.95</td>
      <td>3001.70</td>
      <td>2976.16</td>
      <td>5.815700e+08</td>
      <td>-0.0046</td>
      <td>1002.81</td>
      <td>1001.11</td>
      <td>1007.73</td>
      <td>996.85</td>
      <td>1.080000e+09</td>
      <td>-0.0031</td>
    </tr>
  </tbody>
</table>
<p>247 rows × 15 columns</p>
</div>


- DOW 정보 추가



```python
DJI = fdr.DataReader('dji', start_date, end_date).reset_index()
DJI.name = 'dji'
```


```python
col_rename(DJI)
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
      <th>dji_Close</th>
      <th>dji_Open</th>
      <th>dji_High</th>
      <th>dji_Low</th>
      <th>dji_Volume</th>
      <th>dji_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>30223.89</td>
      <td>30627.47</td>
      <td>30674.28</td>
      <td>29881.82</td>
      <td>476730000.0</td>
      <td>-0.0125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>30391.60</td>
      <td>30204.25</td>
      <td>30504.89</td>
      <td>30141.78</td>
      <td>350910000.0</td>
      <td>0.0055</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>30829.40</td>
      <td>30362.78</td>
      <td>31022.65</td>
      <td>30313.07</td>
      <td>500430000.0</td>
      <td>0.0144</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>31041.13</td>
      <td>30901.18</td>
      <td>31193.40</td>
      <td>30897.86</td>
      <td>430620000.0</td>
      <td>0.0069</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>31097.97</td>
      <td>31069.58</td>
      <td>31140.67</td>
      <td>30793.27</td>
      <td>385650000.0</td>
      <td>0.0018</td>
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
      <th>235</th>
      <td>2021-12-08</td>
      <td>35754.09</td>
      <td>35716.85</td>
      <td>35839.72</td>
      <td>35603.95</td>
      <td>363110000.0</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2021-12-09</td>
      <td>35755.28</td>
      <td>35722.26</td>
      <td>35864.22</td>
      <td>35579.91</td>
      <td>335370000.0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>237</th>
      <td>2021-12-10</td>
      <td>35971.98</td>
      <td>35830.55</td>
      <td>35982.69</td>
      <td>35711.90</td>
      <td>344050000.0</td>
      <td>0.0061</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2021-12-13</td>
      <td>35652.07</td>
      <td>35906.91</td>
      <td>35951.28</td>
      <td>35610.42</td>
      <td>419860000.0</td>
      <td>-0.0089</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2021-12-14</td>
      <td>35545.69</td>
      <td>35605.73</td>
      <td>35779.20</td>
      <td>35441.74</td>
      <td>414320000.0</td>
      <td>-0.0030</td>
    </tr>
  </tbody>
</table>
<p>240 rows × 7 columns</p>
</div>



```python
data = pd.merge(data, DJI, how = 'outer')
```

- 원/달러 환율 정보 추가



```python
EXC = fdr.DataReader('USD/KRW', start_date, end_date).reset_index()
EXC.name = 'exc'
```


```python
col_rename(EXC)
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
      <th>exc_Close</th>
      <th>exc_Open</th>
      <th>exc_High</th>
      <th>exc_Low</th>
      <th>exc_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>1086.48</td>
      <td>1085.73</td>
      <td>1088.30</td>
      <td>1080.02</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>1086.42</td>
      <td>1086.69</td>
      <td>1090.33</td>
      <td>1082.04</td>
      <td>-0.0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>1087.93</td>
      <td>1087.40</td>
      <td>1089.79</td>
      <td>1083.91</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>1094.28</td>
      <td>1088.03</td>
      <td>1096.78</td>
      <td>1085.42</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>1092.93</td>
      <td>1094.35</td>
      <td>1099.21</td>
      <td>1088.79</td>
      <td>-0.0012</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2021-12-08</td>
      <td>1175.19</td>
      <td>1176.85</td>
      <td>1179.31</td>
      <td>1175.12</td>
      <td>-0.0013</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2021-12-09</td>
      <td>1178.15</td>
      <td>1173.42</td>
      <td>1179.37</td>
      <td>1172.62</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2021-12-10</td>
      <td>1180.86</td>
      <td>1178.27</td>
      <td>1182.82</td>
      <td>1176.40</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2021-12-13</td>
      <td>1184.91</td>
      <td>1180.96</td>
      <td>1186.16</td>
      <td>1177.03</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2021-12-14</td>
      <td>1185.19</td>
      <td>1185.10</td>
      <td>1186.38</td>
      <td>1180.97</td>
      <td>0.0002</td>
    </tr>
  </tbody>
</table>
<p>247 rows × 6 columns</p>
</div>



```python
data = pd.merge(data, EXC, how = 'outer')
```


```python
data.columns
```

<pre>
Index(['Date', 'weekday', 'weeknum', 'kospi_Close', 'kospi_Open', 'kospi_High',
       'kospi_Low', 'kospi_Volume', 'kospi_Change', 'kosdaq_Close',
       'kosdaq_Open', 'kosdaq_High', 'kosdaq_Low', 'kosdaq_Volume',
       'kosdaq_Change', 'dji_Close', 'dji_Open', 'dji_High', 'dji_Low',
       'dji_Volume', 'dji_Change', 'exc_Close', 'exc_Open', 'exc_High',
       'exc_Low', 'exc_Change'],
      dtype='object')
</pre>

```python
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
      <th>Date</th>
      <th>weekday</th>
      <th>weeknum</th>
      <th>kospi_Close</th>
      <th>kospi_Open</th>
      <th>kospi_High</th>
      <th>kospi_Low</th>
      <th>kospi_Volume</th>
      <th>kospi_Change</th>
      <th>kosdaq_Close</th>
      <th>...</th>
      <th>dji_Open</th>
      <th>dji_High</th>
      <th>dji_Low</th>
      <th>dji_Volume</th>
      <th>dji_Change</th>
      <th>exc_Close</th>
      <th>exc_Open</th>
      <th>exc_High</th>
      <th>exc_Low</th>
      <th>exc_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-04</td>
      <td>0</td>
      <td>01</td>
      <td>2944.45</td>
      <td>2874.50</td>
      <td>2946.54</td>
      <td>2869.11</td>
      <td>1.030000e+09</td>
      <td>0.0247</td>
      <td>977.62</td>
      <td>...</td>
      <td>30627.47</td>
      <td>30674.28</td>
      <td>29881.82</td>
      <td>476730000.0</td>
      <td>-0.0125</td>
      <td>1086.48</td>
      <td>1085.73</td>
      <td>1088.30</td>
      <td>1080.02</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-05</td>
      <td>1</td>
      <td>01</td>
      <td>2990.57</td>
      <td>2943.67</td>
      <td>2990.57</td>
      <td>2921.84</td>
      <td>1.520000e+09</td>
      <td>0.0157</td>
      <td>985.76</td>
      <td>...</td>
      <td>30204.25</td>
      <td>30504.89</td>
      <td>30141.78</td>
      <td>350910000.0</td>
      <td>0.0055</td>
      <td>1086.42</td>
      <td>1086.69</td>
      <td>1090.33</td>
      <td>1082.04</td>
      <td>-0.0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-06</td>
      <td>2</td>
      <td>01</td>
      <td>2968.21</td>
      <td>2993.34</td>
      <td>3027.16</td>
      <td>2961.37</td>
      <td>1.790000e+09</td>
      <td>-0.0075</td>
      <td>981.39</td>
      <td>...</td>
      <td>30362.78</td>
      <td>31022.65</td>
      <td>30313.07</td>
      <td>500430000.0</td>
      <td>0.0144</td>
      <td>1087.93</td>
      <td>1087.40</td>
      <td>1089.79</td>
      <td>1083.91</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-07</td>
      <td>3</td>
      <td>01</td>
      <td>3031.68</td>
      <td>2980.75</td>
      <td>3055.28</td>
      <td>2980.75</td>
      <td>1.520000e+09</td>
      <td>0.0214</td>
      <td>988.86</td>
      <td>...</td>
      <td>30901.18</td>
      <td>31193.40</td>
      <td>30897.86</td>
      <td>430620000.0</td>
      <td>0.0069</td>
      <td>1094.28</td>
      <td>1088.03</td>
      <td>1096.78</td>
      <td>1085.42</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-08</td>
      <td>4</td>
      <td>01</td>
      <td>3152.18</td>
      <td>3040.11</td>
      <td>3161.11</td>
      <td>3040.11</td>
      <td>1.300000e+09</td>
      <td>0.0397</td>
      <td>987.79</td>
      <td>...</td>
      <td>31069.58</td>
      <td>31140.67</td>
      <td>30793.27</td>
      <td>385650000.0</td>
      <td>0.0018</td>
      <td>1092.93</td>
      <td>1094.35</td>
      <td>1099.21</td>
      <td>1088.79</td>
      <td>-0.0012</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2021-12-08</td>
      <td>2</td>
      <td>49</td>
      <td>3001.80</td>
      <td>3017.93</td>
      <td>3036.13</td>
      <td>2995.34</td>
      <td>4.882500e+08</td>
      <td>0.0034</td>
      <td>1006.04</td>
      <td>...</td>
      <td>35716.85</td>
      <td>35839.72</td>
      <td>35603.95</td>
      <td>363110000.0</td>
      <td>0.0009</td>
      <td>1175.19</td>
      <td>1176.85</td>
      <td>1179.31</td>
      <td>1175.12</td>
      <td>-0.0013</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2021-12-09</td>
      <td>3</td>
      <td>49</td>
      <td>3029.57</td>
      <td>3007.00</td>
      <td>3029.57</td>
      <td>3001.55</td>
      <td>5.134400e+08</td>
      <td>0.0093</td>
      <td>1022.87</td>
      <td>...</td>
      <td>35722.26</td>
      <td>35864.22</td>
      <td>35579.91</td>
      <td>335370000.0</td>
      <td>0.0000</td>
      <td>1178.15</td>
      <td>1173.42</td>
      <td>1179.37</td>
      <td>1172.62</td>
      <td>0.0025</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2021-12-10</td>
      <td>4</td>
      <td>49</td>
      <td>3010.23</td>
      <td>3008.70</td>
      <td>3017.64</td>
      <td>2998.29</td>
      <td>4.516000e+08</td>
      <td>-0.0064</td>
      <td>1011.57</td>
      <td>...</td>
      <td>35830.55</td>
      <td>35982.69</td>
      <td>35711.90</td>
      <td>344050000.0</td>
      <td>0.0061</td>
      <td>1180.86</td>
      <td>1178.27</td>
      <td>1182.82</td>
      <td>1176.40</td>
      <td>0.0023</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2021-12-13</td>
      <td>0</td>
      <td>50</td>
      <td>3001.66</td>
      <td>3019.67</td>
      <td>3043.83</td>
      <td>3000.51</td>
      <td>3.758300e+08</td>
      <td>-0.0028</td>
      <td>1005.96</td>
      <td>...</td>
      <td>35906.91</td>
      <td>35951.28</td>
      <td>35610.42</td>
      <td>419860000.0</td>
      <td>-0.0089</td>
      <td>1184.91</td>
      <td>1180.96</td>
      <td>1186.16</td>
      <td>1177.03</td>
      <td>0.0034</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2021-12-14</td>
      <td>1</td>
      <td>50</td>
      <td>2987.95</td>
      <td>2983.95</td>
      <td>3001.70</td>
      <td>2976.16</td>
      <td>5.815700e+08</td>
      <td>-0.0046</td>
      <td>1002.81</td>
      <td>...</td>
      <td>35605.73</td>
      <td>35779.20</td>
      <td>35441.74</td>
      <td>414320000.0</td>
      <td>-0.0030</td>
      <td>1185.19</td>
      <td>1185.10</td>
      <td>1186.38</td>
      <td>1180.97</td>
      <td>0.0002</td>
    </tr>
  </tbody>
</table>
<p>247 rows × 26 columns</p>
</div>



```python
# data[['kospi_Close', 'kospi_Open', 'kospi_High',
#        'kospi_Low', 'kospi_Volume', 'kospi_Change', 'kosdaq_Close',
#        'kosdaq_Open', 'kosdaq_High', 'kosdaq_Low', 'kosdaq_Volume',
#        'kosdaq_Change', 'dji_Close', 'dji_Open', 'dji_High', 'dji_Low',
#        'dji_Volume', 'dji_Change', 'exc_Close', 'exc_Open', 'exc_High',
#        'exc_Low', 'exc_Change']] = data[['kospi_Close', 'kospi_Open', 'kospi_High',
#        'kospi_Low', 'kospi_Volume', 'kospi_Change', 'kosdaq_Close',
#        'kosdaq_Open', 'kosdaq_High', 'kosdaq_Low', 'kosdaq_Volume',
#        'kosdaq_Change', 'dji_Close', 'dji_Open', 'dji_High', 'dji_Low',
#        'dji_Volume', 'dji_Change', 'exc_Close', 'exc_Open', 'exc_High',
#        'exc_Low', 'exc_Change']].shift(5)
```


```python
data.isnull().sum()
```

<pre>
Date              0
weekday           0
weeknum           0
kospi_Close      11
kospi_Open       11
kospi_High       11
kospi_Low        11
kospi_Volume     11
kospi_Change     11
kosdaq_Close     11
kosdaq_Open      11
kosdaq_High      11
kosdaq_Low       11
kosdaq_Volume    11
kosdaq_Change    11
dji_Close         7
dji_Open          7
dji_High          7
dji_Low           7
dji_Volume        7
dji_Change        7
exc_Close         0
exc_Open          0
exc_High          0
exc_Low           0
exc_Change        0
dtype: int64
</pre>
- 미국 휴장일은 전일 데이터로 fill up



```python
dji_nan = data[data['dji_Open'].isnull()].index
```


```python
data.columns
```

<pre>
Index(['Date', 'weekday', 'weeknum', 'kospi_Close', 'kospi_Open', 'kospi_High',
       'kospi_Low', 'kospi_Volume', 'kospi_Change', 'kosdaq_Close',
       'kosdaq_Open', 'kosdaq_High', 'kosdaq_Low', 'kosdaq_Volume',
       'kosdaq_Change', 'dji_Close', 'dji_Open', 'dji_High', 'dji_Low',
       'dji_Volume', 'dji_Change', 'exc_Close', 'exc_Open', 'exc_High',
       'exc_Low', 'exc_Change'],
      dtype='object')
</pre>

```python
for i in dji_nan:
    for j in range(6,12):
        n = data.iloc[i-1,-j]
        data.iloc[i,-j] = n
```

- 미국 지수의 경우 전일 시가는 24:00전에 알 수 있지만 다른정보는 24:00 이전에 알 수 없으므로 하루를 밀어줌.



```python
data[['dji_Close', 'dji_High', 'dji_Low',
       'dji_Volume', 'dji_Change']] = data[['dji_Close', 'dji_High', 'dji_Low',
       'dji_Volume', 'dji_Change']].shift(1)
```

- 한국 휴장 일은 data 에서 삭제.



```python
kospi_nan = data[data['kospi_Open'].isnull()].index
exc_nan = data[data['exc_Open'].isnull()].index
```


```python
nan_list = set(kospi_nan.to_list() + exc_nan.to_list())
```


```python
data.drop(nan_list, inplace=True)
```


```python
data.drop(0, axis=0, inplace=True)
```


```python
shift_col = data.columns.drop(['Date', 'weekday', 'weeknum'])
```


```python
data[shift_col] = data[shift_col].shift(5)
```


```python
data = data[5:].reset_index(drop=True)
```


```python
data.isnull().sum()
```

<pre>
Date             0
weekday          0
weeknum          0
kospi_Close      0
kospi_Open       0
kospi_High       0
kospi_Low        0
kospi_Volume     0
kospi_Change     0
kosdaq_Close     0
kosdaq_Open      0
kosdaq_High      0
kosdaq_Low       0
kosdaq_Volume    0
kosdaq_Change    0
dji_Close        0
dji_Open         0
dji_High         0
dji_Low          0
dji_Volume       0
dji_Change       0
exc_Close        0
exc_Open         0
exc_High         0
exc_Low          0
exc_Change       0
dtype: int64
</pre>

```python
ms_data = data.copy()
```


```python
data['weekday'] = data['weekday'].astype('str')
```


```python
dummies = pd.get_dummies(data[['weekday', 'weeknum']], prefix=['weekday','weeknum'])
```


```python
m_data = pd.concat([data, dummies], axis=1)
```


```python
m_data.drop(['weekday','weeknum'], axis=1, inplace=True)
```


```python
m_data
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
      <th>kospi_Close</th>
      <th>kospi_Open</th>
      <th>kospi_High</th>
      <th>kospi_Low</th>
      <th>kospi_Volume</th>
      <th>kospi_Change</th>
      <th>kosdaq_Close</th>
      <th>kosdaq_Open</th>
      <th>kosdaq_High</th>
      <th>...</th>
      <th>weeknum_41</th>
      <th>weeknum_42</th>
      <th>weeknum_43</th>
      <th>weeknum_44</th>
      <th>weeknum_45</th>
      <th>weeknum_46</th>
      <th>weeknum_47</th>
      <th>weeknum_48</th>
      <th>weeknum_49</th>
      <th>weeknum_50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-12</td>
      <td>2990.57</td>
      <td>2943.67</td>
      <td>2990.57</td>
      <td>2921.84</td>
      <td>1.520000e+09</td>
      <td>0.0157</td>
      <td>985.76</td>
      <td>976.43</td>
      <td>985.76</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-13</td>
      <td>2968.21</td>
      <td>2993.34</td>
      <td>3027.16</td>
      <td>2961.37</td>
      <td>1.790000e+09</td>
      <td>-0.0075</td>
      <td>981.39</td>
      <td>987.25</td>
      <td>990.88</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-14</td>
      <td>3031.68</td>
      <td>2980.75</td>
      <td>3055.28</td>
      <td>2980.75</td>
      <td>1.520000e+09</td>
      <td>0.0214</td>
      <td>988.86</td>
      <td>983.28</td>
      <td>993.91</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-15</td>
      <td>3152.18</td>
      <td>3040.11</td>
      <td>3161.11</td>
      <td>3040.11</td>
      <td>1.300000e+09</td>
      <td>0.0397</td>
      <td>987.79</td>
      <td>990.70</td>
      <td>995.22</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-18</td>
      <td>3148.45</td>
      <td>3161.90</td>
      <td>3266.23</td>
      <td>3096.19</td>
      <td>1.710000e+09</td>
      <td>-0.0012</td>
      <td>976.63</td>
      <td>988.38</td>
      <td>993.20</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>2021-12-08</td>
      <td>2899.72</td>
      <td>2860.12</td>
      <td>2905.74</td>
      <td>2837.03</td>
      <td>5.639300e+08</td>
      <td>0.0214</td>
      <td>977.15</td>
      <td>969.90</td>
      <td>977.73</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226</th>
      <td>2021-12-09</td>
      <td>2945.27</td>
      <td>2874.64</td>
      <td>2945.27</td>
      <td>2874.64</td>
      <td>5.344600e+08</td>
      <td>0.0157</td>
      <td>977.43</td>
      <td>967.34</td>
      <td>978.16</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227</th>
      <td>2021-12-10</td>
      <td>2968.33</td>
      <td>2935.93</td>
      <td>2975.44</td>
      <td>2927.55</td>
      <td>4.867500e+08</td>
      <td>0.0078</td>
      <td>998.47</td>
      <td>981.65</td>
      <td>998.49</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>228</th>
      <td>2021-12-13</td>
      <td>2973.25</td>
      <td>2954.82</td>
      <td>2983.50</td>
      <td>2932.49</td>
      <td>4.799100e+08</td>
      <td>0.0017</td>
      <td>991.87</td>
      <td>990.07</td>
      <td>994.15</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>229</th>
      <td>2021-12-14</td>
      <td>2991.72</td>
      <td>2973.84</td>
      <td>2992.31</td>
      <td>2960.90</td>
      <td>5.413700e+08</td>
      <td>0.0062</td>
      <td>996.64</td>
      <td>996.89</td>
      <td>999.52</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 78 columns</p>
</div>


### 옵션/선물 만기일



```python
# https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&qvt=0&query=%EC%98%B5%EC%85%98%EB%A7%8C%EA%B8%B0%EC%9D%BC
f_option = ['2021-01-14', '2021-02-10', '2021-03-11', '2021-04-08', '2021-05-13', '2021-06-10', 
            '2021-07-08', '2021-08-12', '2021-09-09', '2021-10-14', '2021-11-11', '2021-12-09']
```


```python
f_futures = ['2021-03-11', '2021-06-10', '2021-09-09', '2021-12-09']
```

### 전체 모델링



```python
sample_name = 'sample_submission.csv'
sample_submission = pd.read_csv(os.path.join(path, sample_name))
```


```python
submission = sample_submission.set_index('Day')
```


```python
from xgboost import XGBRegressor
```


```python
xgb = XGBRegressor()
```


```python
day_list = ['2021-11-29', '2021-11-30', '2021-12-01', '2021-12-02', '2021-12-03']
```


```python
stock_list.drop(stock_list[stock_list['종목코드']=='031390'].index,inplace=True)
```


```python
for code in tqdm(stock_list['종목코드'].values):
    for day in day_list:
        code_data = fdr.DataReader(code, start = start_date, end = end_date).reset_index()
        code_data['y_Close'] = code_data['Close']    
        
        code_data[[
            'Open', 'High', 'Low', 'Close', 'Volume', 'Change']] = code_data[[
            'Open', 'High', 'Low', 'Close', 'Volume', 'Change']].shift(5)

        #이동 평균선 추가 (5/10/20)
        code_data['5_close'] = code_data['Close'].rolling(window=5).mean()   
        code_data['10_close'] = code_data['Close'].rolling(window=10).mean()   
        code_data['20_close'] = code_data['Close'].rolling(window=20).mean()   
        
        #6일전/7일전 종가 추가
        code_data['6d_Close'] = code_data[['Close']].shift(1)
        code_data['7d_Close'] = code_data[['6d_Close']].shift(1)
        
        #6일전/7일전 종가와 전일 종가의 차이 비율 추가
        code_data['diff_Close'] = (code_data['Close'] - code_data['6d_Close'])/code_data['Close']
        code_data['diff_Close2'] = (code_data['Close'] - code_data['7d_Close'])/code_data['Close']

        
        #6일전 거래량 추가
        code_data['6d_Volume'] = code_data[['Volume']].shift(6)
        
        #2일전 거래량과 전일 거래량의 차이 추가
        code_data['diff_Volume'] = code_data['Volume'] - code_data['6d_Volume']
               
        df = pd.merge(m_data, code_data, how='left', on='Date')
        df = df.drop(df[df.isnull().any(axis=1)].index, axis=0)
       
        market = stock_list[stock_list['종목코드'].str.contains(code)]['상장시장'].values[0]
       
        if market == 'KOSPI':
            df_kospi = df.drop(['kosdaq_Close', 'kosdaq_Open', 'kosdaq_High', 'kosdaq_Low', 'kosdaq_Volume', 'kosdaq_Change'], axis=1)
            x_train = df_kospi[df_kospi['Date'] < day].drop(['Date','y_Close'], axis=1)
            y_train = df_kospi[df_kospi['Date'] < day]['y_Close']
            xgb.fit(x_train, y_train)
            pred = xgb.predict(df_kospi[df_kospi['Date']==day].drop(['Date','y_Close'], axis=1))
        else:
            df_kosdaq = df.drop(['kospi_Close', 'kospi_Open', 'kospi_High', 'kospi_Low','kospi_Volume', 'kospi_Change', 'kosdaq_Close', 'kosdaq_Open'], axis=1)
            x_train = df_kosdaq[df_kosdaq['Date'] < day].drop(['Date','y_Close'], axis=1)
            y_train = df_kosdaq[df_kosdaq['Date'] < day]['y_Close']
            xgb.fit(x_train, y_train)
            pred = xgb.predict(df_kosdaq[df_kosdaq['Date']==day].drop(['Date','y_Close'], axis=1))
        submission.loc[day,code] = pred
```

<pre>
100%|████████████████████████████████████████████████████████████████████████████████| 369/369 [06:12<00:00,  1.01s/it]
</pre>

```python
# privite submission만 데이터셋으로 작성
```


```python
submission = submission.iloc[5:,:]
submission
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
      <th>000060</th>
      <th>000080</th>
      <th>000100</th>
      <th>000120</th>
      <th>000150</th>
      <th>000240</th>
      <th>000250</th>
      <th>000270</th>
      <th>000660</th>
      <th>000670</th>
      <th>...</th>
      <th>330860</th>
      <th>336260</th>
      <th>336370</th>
      <th>347860</th>
      <th>348150</th>
      <th>348210</th>
      <th>352820</th>
      <th>357780</th>
      <th>363280</th>
      <th>950130</th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-11-29</th>
      <td>29417.746094</td>
      <td>33234.343750</td>
      <td>60132.703125</td>
      <td>136056.812500</td>
      <td>119314.929688</td>
      <td>16740.277344</td>
      <td>45884.242188</td>
      <td>82530.968750</td>
      <td>107088.679688</td>
      <td>683745.0625</td>
      <td>...</td>
      <td>46994.058594</td>
      <td>49025.132812</td>
      <td>89673.695312</td>
      <td>37659.074219</td>
      <td>19519.048828</td>
      <td>50946.542969</td>
      <td>372958.59375</td>
      <td>265210.156250</td>
      <td>25597.248047</td>
      <td>16075.208008</td>
    </tr>
    <tr>
      <th>2021-11-30</th>
      <td>30002.962891</td>
      <td>30610.638672</td>
      <td>59678.476562</td>
      <td>129827.921875</td>
      <td>117337.351562</td>
      <td>15388.730469</td>
      <td>44593.121094</td>
      <td>79988.164062</td>
      <td>114570.687500</td>
      <td>653516.1875</td>
      <td>...</td>
      <td>43664.417969</td>
      <td>50585.960938</td>
      <td>97681.562500</td>
      <td>35288.453125</td>
      <td>20594.652344</td>
      <td>50690.828125</td>
      <td>379092.87500</td>
      <td>264948.375000</td>
      <td>23948.753906</td>
      <td>16622.246094</td>
    </tr>
    <tr>
      <th>2021-12-01</th>
      <td>29913.453125</td>
      <td>29738.423828</td>
      <td>58936.699219</td>
      <td>125720.078125</td>
      <td>116199.320312</td>
      <td>15408.984375</td>
      <td>45524.351562</td>
      <td>79979.945312</td>
      <td>115808.617188</td>
      <td>643901.2500</td>
      <td>...</td>
      <td>44753.972656</td>
      <td>49181.937500</td>
      <td>97690.156250</td>
      <td>36902.507812</td>
      <td>19222.310547</td>
      <td>46054.750000</td>
      <td>367844.65625</td>
      <td>259253.796875</td>
      <td>23623.125000</td>
      <td>17566.416016</td>
    </tr>
    <tr>
      <th>2021-12-02</th>
      <td>30087.794922</td>
      <td>30333.685547</td>
      <td>57824.656250</td>
      <td>126370.695312</td>
      <td>114069.851562</td>
      <td>15501.014648</td>
      <td>44904.937500</td>
      <td>80623.687500</td>
      <td>116014.000000</td>
      <td>648058.5000</td>
      <td>...</td>
      <td>46721.980469</td>
      <td>49454.167969</td>
      <td>100551.687500</td>
      <td>36566.273438</td>
      <td>20361.396484</td>
      <td>48312.214844</td>
      <td>375545.75000</td>
      <td>272749.937500</td>
      <td>24086.357422</td>
      <td>16994.531250</td>
    </tr>
    <tr>
      <th>2021-12-03</th>
      <td>31241.271484</td>
      <td>29993.458984</td>
      <td>60009.445312</td>
      <td>128049.921875</td>
      <td>116856.796875</td>
      <td>16200.601562</td>
      <td>44051.550781</td>
      <td>80780.773438</td>
      <td>115700.054688</td>
      <td>662439.3750</td>
      <td>...</td>
      <td>48972.601562</td>
      <td>46961.933594</td>
      <td>98532.125000</td>
      <td>37583.468750</td>
      <td>21500.310547</td>
      <td>49715.628906</td>
      <td>343600.87500</td>
      <td>269958.593750</td>
      <td>25039.224609</td>
      <td>17610.996094</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 370 columns</p>
</div>


### 실제 종가와 비교



```python
true_close = sample_submission.copy().set_index('Day')

code_data = fdr.DataReader('000060', start = '2021-11-01', end = '2021-11-01')['Close'].reset_index()
```


```python
for code in tqdm(stock_list['종목코드'].values):
    for day in day_list:
        code_data = fdr.DataReader(code, start = day, end = day)['Close'].reset_index()
        true_close.loc[day,code] = code_data['Close'].values[0]
```

<pre>
100%|████████████████████████████████████████████████████████████████████████████████| 369/369 [02:19<00:00,  2.65it/s]
</pre>

```python
true_close = true_close.iloc[5:,:]
```


```python
true_close
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
      <th>000060</th>
      <th>000080</th>
      <th>000100</th>
      <th>000120</th>
      <th>000150</th>
      <th>000240</th>
      <th>000250</th>
      <th>000270</th>
      <th>000660</th>
      <th>000670</th>
      <th>...</th>
      <th>330860</th>
      <th>336260</th>
      <th>336370</th>
      <th>347860</th>
      <th>348150</th>
      <th>348210</th>
      <th>352820</th>
      <th>357780</th>
      <th>363280</th>
      <th>950130</th>
    </tr>
    <tr>
      <th>Day</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-11-29</th>
      <td>31200.0</td>
      <td>30300.0</td>
      <td>58900.0</td>
      <td>129000.0</td>
      <td>110500.0</td>
      <td>15200.0</td>
      <td>43250.0</td>
      <td>79200.0</td>
      <td>116000.0</td>
      <td>641000.0</td>
      <td>...</td>
      <td>43750.0</td>
      <td>48750.0</td>
      <td>100500.0</td>
      <td>37100.0</td>
      <td>19800.0</td>
      <td>49100.0</td>
      <td>369000.0</td>
      <td>266300.0</td>
      <td>24200.0</td>
      <td>17650.0</td>
    </tr>
    <tr>
      <th>2021-11-30</th>
      <td>31300.0</td>
      <td>29000.0</td>
      <td>57800.0</td>
      <td>124000.0</td>
      <td>108500.0</td>
      <td>15150.0</td>
      <td>42150.0</td>
      <td>77800.0</td>
      <td>114000.0</td>
      <td>632000.0</td>
      <td>...</td>
      <td>43300.0</td>
      <td>49250.0</td>
      <td>97800.0</td>
      <td>35550.0</td>
      <td>18600.0</td>
      <td>45500.0</td>
      <td>364500.0</td>
      <td>255800.0</td>
      <td>23100.0</td>
      <td>19500.0</td>
    </tr>
    <tr>
      <th>2021-12-01</th>
      <td>31700.0</td>
      <td>29400.0</td>
      <td>57700.0</td>
      <td>125000.0</td>
      <td>112000.0</td>
      <td>15500.0</td>
      <td>42750.0</td>
      <td>81200.0</td>
      <td>116500.0</td>
      <td>630000.0</td>
      <td>...</td>
      <td>48950.0</td>
      <td>48700.0</td>
      <td>98500.0</td>
      <td>36050.0</td>
      <td>18900.0</td>
      <td>46200.0</td>
      <td>352500.0</td>
      <td>264200.0</td>
      <td>23850.0</td>
      <td>18650.0</td>
    </tr>
    <tr>
      <th>2021-12-02</th>
      <td>32150.0</td>
      <td>29550.0</td>
      <td>60100.0</td>
      <td>129000.0</td>
      <td>110000.0</td>
      <td>16150.0</td>
      <td>43100.0</td>
      <td>81600.0</td>
      <td>120000.0</td>
      <td>639000.0</td>
      <td>...</td>
      <td>51900.0</td>
      <td>46250.0</td>
      <td>95100.0</td>
      <td>33800.0</td>
      <td>18650.0</td>
      <td>48550.0</td>
      <td>330000.0</td>
      <td>274700.0</td>
      <td>25200.0</td>
      <td>18050.0</td>
    </tr>
    <tr>
      <th>2021-12-03</th>
      <td>32700.0</td>
      <td>30600.0</td>
      <td>60300.0</td>
      <td>131000.0</td>
      <td>108500.0</td>
      <td>16400.0</td>
      <td>44900.0</td>
      <td>82500.0</td>
      <td>118000.0</td>
      <td>642000.0</td>
      <td>...</td>
      <td>51900.0</td>
      <td>46800.0</td>
      <td>95300.0</td>
      <td>34500.0</td>
      <td>19100.0</td>
      <td>49000.0</td>
      <td>354500.0</td>
      <td>275900.0</td>
      <td>25800.0</td>
      <td>18150.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 370 columns</p>
</div>



```python
diff = np.mean(np.abs(submission-true_close)/true_close)
```


```python
np.mean(np.mean(np.abs(submission[:5] - true_close[:5])/true_close[:5]))*100
```

<pre>
4.453063727451483
</pre>

```python
```
