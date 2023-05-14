---
layout: single
title:  "DACON 제주도 도로 교통량 예측 "
categories: Competitions
# tag: [data science, 대회]
toc: true
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


22년 11월 진행된 Dacon 제주도 도로 교통량 예측 대회의 코드를 공유합니다.

(대회 정보 : https://dacon.io/competitions/official/235985/overview/description)


## Import



```python
import pandas as pd
import numpy as np

from workalendar.asia import SouthKorea
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gc

import warnings
warnings.filterwarnings(action='ignore')
```

## Data Load



```python
def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'../origin_data/{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')
```


```python
csv_to_parquet('../origin_data/train.csv', 'train')
csv_to_parquet('../origin_data/test.csv', 'test')
```

<pre>
train Done.
test Done.
</pre>

```python
train = pd.read_parquet('../origin_data/train.parquet')
test = pd.read_parquet('../origin_data/test.parquet')
```

## EDA



```python
for i in train['maximum_speed_limit'].unique():
    idx = np.where(train['maximum_speed_limit']==i)
    sns.kdeplot(train.loc[idx, 'target'], label=i)
plt.legend()
plt.show()
```

- Limit Speed 별로 target이 다른 분포를 보임.



```python
gps = train[['start_longitude', 'end_longitude', 'start_latitude', 'end_latitude', 'target']]
```


```python
gps_set = [gps['start_longitude'].min(), gps['start_longitude'].max(), gps['start_latitude'].min(), gps['start_latitude'].max()] # 지도 그림의 gps 좌표
gps_set
```

<pre>
[126.182616549771, 126.930940973848, 33.2434317486804, 33.5560801767072]
</pre>

```python
vel_low_idx = gps.loc[gps['target']<15].index # 시내 교통 체증기준 10 km/h 미만
vel_high_idx = gps.loc[gps['target']>80].index # 고속도로 원활기준 80 km/h 초과
```

- GPS 좌표를 기준으로 시각화  

(참고자료 : https://towardsdatascience.com/simple-gps-data-visualization-using-python-and-open-street-maps-50f992e9b676)



```python
f, ax = plt.subplots(figsize=(21,10))

ax.set_xlim(gps_set[0], gps_set[1])
ax.set_ylim(gps_set[2], gps_set[3])

image = plt.imread('map.png')
ax.imshow(image, zorder=0, extent=gps_set, aspect='equal')

for i in tqdm(vel_low_idx): # 교통 체증 도로 빨강
    x_1 = gps.loc[i,'start_longitude']
    x_2 = gps.loc[i,'end_longitude'] 
    y_1 = gps.loc[i,'start_latitude']
    y_2 = gps.loc[i,'end_latitude'] 
    ax.plot([x_1, x_2], [y_1, y_2], color='red')

for i in tqdm(vel_high_idx): # 교통 원활 도로 파랑
    x_1 = gps.loc[i,'start_longitude']
    x_2 = gps.loc[i,'end_longitude'] 
    y_1 = gps.loc[i,'start_latitude']
    y_2 = gps.loc[i,'end_latitude'] 
    ax.plot([x_1, x_2], [y_1, y_2], color='blue')

plt.show()
```

<pre>
100%|██████████| 111725/111725 [00:45<00:00, 2475.80it/s]
100%|██████████| 26239/26239 [00:08<00:00, 2942.93it/s]
</pre>

- 가장 Importance가 큰 Max Speed Limit를 기준으로 Data Set을 분할



```python
for i in range(30,90,10):
    if i==40: # Test Data에 max speed limit 40은 없으므로 생략.
        continue
    else:
        globals()[f'train_{i}'] = train.loc[train['maximum_speed_limit']==i]
```

- Data set 별로 모델 Tuning. (Data set 별 Tuning 코드는 생략)



```python
parameters = {'max_depth': np.arange(7,13,1),
                # 'n_estimators' : np.arange(10,101,10),
                # 'gamma' : np.arange(0,10,1),
              'random_state': [random_state]
              }

estimator = XGBRegressor()
xgb = GridSearchCV(estimator=estimator, 
                   param_grid=parameters, 
                   scoring='neg_mean_absolute_error', 
                   cv=5,
                   verbose=3)
xgb.fit(train_80.drop('target',axis=1), train_80['target'])
print(xgb.best_estimator_)
```

<pre>
Fitting 5 folds for each of 6 candidates, totalling 30 fits
[CV 1/5] END .....max_depth=7, random_state=42;, score=-2.919 total time=  12.1s
[CV 2/5] END .....max_depth=7, random_state=42;, score=-2.939 total time=  12.2s
[CV 3/5] END .....max_depth=7, random_state=42;, score=-2.935 total time=  12.1s
[CV 4/5] END .....max_depth=7, random_state=42;, score=-2.931 total time=  12.0s
[CV 5/5] END .....max_depth=7, random_state=42;, score=-2.932 total time=  12.0s
[CV 1/5] END .....max_depth=8, random_state=42;, score=-2.865 total time=  14.1s
[CV 2/5] END .....max_depth=8, random_state=42;, score=-2.881 total time=  14.1s
[CV 3/5] END .....max_depth=8, random_state=42;, score=-2.871 total time=  14.1s
[CV 4/5] END .....max_depth=8, random_state=42;, score=-2.871 total time=  14.1s
[CV 5/5] END .....max_depth=8, random_state=42;, score=-2.875 total time=  14.0s
[CV 1/5] END .....max_depth=9, random_state=42;, score=-2.833 total time=  16.3s
[CV 2/5] END .....max_depth=9, random_state=42;, score=-2.846 total time=  16.4s
[CV 3/5] END .....max_depth=9, random_state=42;, score=-2.837 total time=  16.3s
[CV 4/5] END .....max_depth=9, random_state=42;, score=-2.832 total time=  16.2s
[CV 5/5] END .....max_depth=9, random_state=42;, score=-2.838 total time=  16.2s
[CV 1/5] END ....max_depth=10, random_state=42;, score=-2.819 total time=  18.8s
[CV 2/5] END ....max_depth=10, random_state=42;, score=-2.833 total time=  18.6s
[CV 3/5] END ....max_depth=10, random_state=42;, score=-2.819 total time=  18.5s
[CV 4/5] END ....max_depth=10, random_state=42;, score=-2.822 total time=  18.6s
[CV 5/5] END ....max_depth=10, random_state=42;, score=-2.817 total time=  18.6s
[CV 1/5] END ....max_depth=11, random_state=42;, score=-2.827 total time=  21.2s
[CV 2/5] END ....max_depth=11, random_state=42;, score=-2.844 total time=  21.2s
[CV 3/5] END ....max_depth=11, random_state=42;, score=-2.827 total time=  22.0s
[CV 4/5] END ....max_depth=11, random_state=42;, score=-2.830 total time=  21.2s
[CV 5/5] END ....max_depth=11, random_state=42;, score=-2.827 total time=  21.4s
[CV 1/5] END ....max_depth=12, random_state=42;, score=-2.858 total time=  23.9s
[CV 2/5] END ....max_depth=12, random_state=42;, score=-2.868 total time=  24.2s
[CV 3/5] END ....max_depth=12, random_state=42;, score=-2.857 total time=  24.3s
[CV 4/5] END ....max_depth=12, random_state=42;, score=-2.864 total time=  23.9s
[CV 5/5] END ....max_depth=12, random_state=42;, score=-2.857 total time=  23.9s
XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
             importance_type=None, interaction_constraints='',
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=10, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=42,
             reg_alpha=0, reg_lambda=1, ...)
</pre>
## Prediction



```python
for i in range(30,90,10):
    if i==40:
        continue
    else:
        globals()[f'X_train_{i}'] = X_train.loc[X_train['maximum_speed_limit']==i]
```


```python
# 튜팅된 Parameter를 각 모델에 부여
for i in range(30,90,10):
    if i==40:
        continue
    elif i==30:
        n_train = eval(f'X_train_{i}')
        n_y_train = y_train[eval(f'X_train_{i}').index]
        globals()[f'xgb_{i}'] = XGBRegressor(random_state=random_state, max_depth=9).fit(n_train, n_y_train)
    elif i==50:
        n_train = eval(f'X_train_{i}')
        n_y_train = y_train[eval(f'X_train_{i}').index]
        globals()[f'xgb_{i}'] = XGBRegressor(random_state=random_state, max_depth=12).fit(n_train, n_y_train)
    elif i==60:
        n_train = eval(f'X_train_{i}')
        n_y_train = y_train[eval(f'X_train_{i}').index]
        globals()[f'xgb_{i}'] = XGBRegressor(random_state=random_state, max_depth=12).fit(n_train, n_y_train)
    elif i==70:
        n_train = eval(f'X_train_{i}')
        n_y_train = y_train[eval(f'X_train_{i}').index]
        globals()[f'xgb_{i}'] = XGBRegressor(random_state=random_state, max_depth=11).fit(n_train, n_y_train)
    elif i==80:
        n_train = eval(f'X_train_{i}')
        n_y_train = y_train[eval(f'X_train_{i}').index]
        globals()[f'xgb_{i}'] = XGBRegressor(random_state=random_state, max_depth=10).fit(n_train, n_y_train)
```


```python
# Limit Speed 별 모델로 test data set의 target 예측
for i in range(30,90,10):
    if i==40:
        continue
    else:
        globals()[f'pred_xgb_{i}'] = eval(f'xgb_{i}').predict(test.loc[test['maximum_speed_limit']==i])
```


```python
# 모델 예측값을 각각 대입
for i in range(30,90,10):
    if i==40:
        continue
    else:
        test.loc[test['maximum_speed_limit']==i, 'pred'] = eval(f'pred_xgb_{i}')
```


```python
pred = test['pred']
```


```python
sample_submission = pd.read_csv('../origin_data/sample_submission.csv')
```


```python
pred.reset_index(inplace=True, drop=True)
```

## Submission



```python
# Submission 파일 생성
sample_submission['target'] = pred
today= datetime.now().date().strftime('%m%d')
sample_submission.to_csv(f"./99.submission/submit_{today}_1.csv", index = False)
```


```python
sample_submission
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
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000000</td>
      <td>26.753897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_000001</td>
      <td>42.798801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_000002</td>
      <td>67.241051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_000003</td>
      <td>37.343075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_000004</td>
      <td>43.202198</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>291236</th>
      <td>TEST_291236</td>
      <td>47.885696</td>
    </tr>
    <tr>
      <th>291237</th>
      <td>TEST_291237</td>
      <td>51.416042</td>
    </tr>
    <tr>
      <th>291238</th>
      <td>TEST_291238</td>
      <td>23.034536</td>
    </tr>
    <tr>
      <th>291239</th>
      <td>TEST_291239</td>
      <td>23.455151</td>
    </tr>
    <tr>
      <th>291240</th>
      <td>TEST_291240</td>
      <td>48.867462</td>
    </tr>
  </tbody>
</table>
<p>291241 rows × 2 columns</p>
</div>

