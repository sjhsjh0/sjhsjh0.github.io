---
layout: single
title:  "여러개 파일을 변수로 저장하기"
categories: DataSet
# tag: [data science, vision, cs231n]
# use_math: true
toc: true
# toc_sticky: true
author_profile: false
# published: true
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

<br/>

데이터 분석을 하다보면 여러개의 csv, excel 파일 등을 한번에 여러개의 변수로 불러와야 하는 일이 생긴다.  

<br/>









```python
import glob
import pandas as pd
```


<br/>

폴더에 있는 파일을 쉽게 불러오기 위해 glob를 import 해준다.

<br/>



```python
glob.glob("*.csv")
```



<pre>
['test0.csv',
 'test1.csv',
 'test2.csv',
 'test3.csv',
 'test4.csv',
 'test5.csv',
 'test6.csv',
 'test7.csv',
 'test8.csv']
</pre>

<br/>

jupyter notebook 파일이 있는 폴더에는 총 9개의 csv파일이 있다.

<br/>


```python
files = glob.glob("*.csv")
files
```



<pre>
['test0.csv',
 'test1.csv',
 'test2.csv',
 'test3.csv',
 'test4.csv',
 'test5.csv',
 'test6.csv',
 'test7.csv',
 'test8.csv']
</pre>

<br/>

files에 csv 파일명 list를 담아준다.

<br/>


```python
df = pd.read_csv(files[0])
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
      <th>survived</th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Klasen, Mr. Klas Albin</td>
      <td>male</td>
      <td>18.0</td>
      <td>1</td>
      <td>1</td>
      <td>350404</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Compton, Miss. Sara Rebecca</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17756</td>
      <td>83.1583</td>
      <td>E49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Moor, Master. Meier</td>
      <td>male</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>392096</td>
      <td>12.4750</td>
      <td>E121</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Homer, Mr. Harry ("Mr E Haven")</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>111426</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Toufik, Mr. Nakli</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2641</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>


<br/>

pandas의 read_csv를 활용하여 csv파일을 dataframe으로 쉽게 불러 올 수 있다. 

<br/>

<br/>

9개의 파일을 모두 불러와 보자.

<br/>



```python
df_0 = pd.read_csv(files[0])
df_1 = pd.read_csv(files[1])
df_2 = pd.read_csv(files[2])
df_3 = pd.read_csv(files[3])
df_4 = pd.read_csv(files[4])
df_5 = pd.read_csv(files[5])
df_6 = pd.read_csv(files[6])
df_7 = pd.read_csv(files[7])
df_8 = pd.read_csv(files[8])
```

<br/>

변수를 하나씩 지정해주고 파일을 하나씩 각 변수에 담을 수는 있지만 귀찮은 일이다.

<br/>

<br/>

for문을 사용하면 한번에 여러개 변수에 각 csv 파일을 담을 수 있을 것 같아 보인다.

<br/>



```python
for i in range(len(files)):
    str(df) + "_" + str(i) = pd.read_csv(files[0])

  File "<ipython-input-11-26d61a527bfa>", line 2
    df + "_" + str(i) = pd.read_csv(files[i])
    ^
SyntaxError: cannot assign to operator
```

for문으로 변수명을 바꾸면서 파일을 불러와 보려 했지만 위의 코드를 그대로 실행하면 오류가 난다.
<br/><br/>


여러개의 변수를 for문을 활용하여 만들어 주기 위해서는 for문과 함께 globals를 활용할 수 있다. 

<br/><br/>

```python
for i in range(len(files)):
    globals()['df'+'_'+str(i)] = pd.read_csv(files[i])
```
globals를 활용하면 여러개의 변수에 DataFrame을  쉽게 담을 수 있다.

<br/><br/>


```python
df_0.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Klasen, Mr. Klas Albin</td>
      <td>male</td>
      <td>18.0</td>
      <td>1</td>
      <td>1</td>
      <td>350404</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Compton, Miss. Sara Rebecca</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17756</td>
      <td>83.1583</td>
      <td>E49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Moor, Master. Meier</td>
      <td>male</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>392096</td>
      <td>12.4750</td>
      <td>E121</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Homer, Mr. Harry ("Mr E Haven")</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>111426</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Toufik, Mr. Nakli</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2641</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_2.head()
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
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Klasen, Mr. Klas Albin</td>
      <td>male</td>
      <td>18.0</td>
      <td>1</td>
      <td>1</td>
      <td>350404</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Compton, Miss. Sara Rebecca</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17756</td>
      <td>83.1583</td>
      <td>E49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Moor, Master. Meier</td>
      <td>male</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>392096</td>
      <td>12.4750</td>
      <td>E121</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Homer, Mr. Harry ("Mr E Haven")</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>111426</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Toufik, Mr. Nakli</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2641</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



```python
df_7
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
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Klasen, Mr. Klas Albin</td>
      <td>male</td>
      <td>18.0</td>
      <td>1</td>
      <td>1</td>
      <td>350404</td>
      <td>7.8542</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Compton, Miss. Sara Rebecca</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17756</td>
      <td>83.1583</td>
      <td>E49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Moor, Master. Meier</td>
      <td>male</td>
      <td>6.0</td>
      <td>0</td>
      <td>1</td>
      <td>392096</td>
      <td>12.4750</td>
      <td>E121</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Homer, Mr. Harry ("Mr E Haven")</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>111426</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Toufik, Mr. Nakli</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2641</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>Herman, Mrs. Samuel (Jane Laver)</td>
      <td>female</td>
      <td>48.0</td>
      <td>1</td>
      <td>2</td>
      <td>220845</td>
      <td>65.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>Gheorgheff, Mr. Stanio</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349254</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>Vander Cruyssen, Mr. Victor</td>
      <td>male</td>
      <td>47.0</td>
      <td>0</td>
      <td>0</td>
      <td>345765</td>
      <td>9.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>Isham, Miss. Ann Elizabeth</td>
      <td>female</td>
      <td>50.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17595</td>
      <td>28.7125</td>
      <td>C49</td>
      <td>C</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>Lefebre, Miss. Ida</td>
      <td>female</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>4133</td>
      <td>25.4667</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


변수들에 DataFrame이 잘 입력된 것을 볼 수 있다.

