# Microsoft Studios Movie Project


# 1. Introduction

## 1.1 Overview

Films are a major part of popular culture and a huge source of entertainment for many people. The market is projected to reach USD 409.02 billion by 2026 in terms of revenue. Therefore, it is wise to invest in this market. I will use data from various websites that contain information about the film industry.

## 1.2 Objectives

1. Business Understanding<br>
2. Data Understanding <br>
3. Data Preparation
4. Data Analysis
5. Conclusion

# 2. Business Understanding

The tech giant Microsoft has decided to venture into creating original video content and wants to establish a movie studio. My goal is to use exploratory data analysis to produce insights for Microsoft as they enter the film industry. I will be looking for answers to the following questions:

1.What are the most profitable genres?

2.What are the most popular film ratings?

3.Who are the best directors to produce movies?

4.Which studios can be used as a benchmark?

# 3. Data Understanding

The datasets provided for this analysis were collected from different movie review aggregation sites and contain information on the various movie genres and their popularity among critics and viewers.  
The datasets include:
1. [IMDB](https://www.imdb.com/) 
2. [Box Office Mojo](https://www.boxofficemojo.com/) 
3. [Rotten Tomatoes](https://www.rottentomatoes.com/)



```python

```

## Steps
1. Load the data with pandas and explore the dataframes.
2. Clean the data by dealing with:
    - missing values
    - duplicate rows
    - invalid data
    - outliers
3. Perform exploratory analysis in order to answer the business questions.
4. Conclusion.
5. Recommendations.

# 4. Loading Libraries and Datasets


```python
#Importing the packages I will be using for this project
import numpy as np
import sqlite3
import pandas as pd
import zipfile
import csv
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
#loading the datasets
#first extract the im.db file and create a database connection
zf = zipfile.ZipFile('zippedData/im.db.zip')
zf.extract('im.db')
conn = sqlite3.connect('im.db')
#Loading other datasets
rt_reviews = pd.read_csv('zippedData/rt.reviews.tsv.gz',delimiter = "\t",encoding='latin-1')
rt_movies = pd.read_csv('zippedData/rt.movie_info.tsv.gz',delimiter = '\t')
bom_movies = pd.read_csv('zippedData/bom.movie_gross.csv.gz')
movie_budgets1 = pd.read_csv('zippedData/tn.movie_budgets.csv.gz')

```

# 6. Previewing the Datasets

#### a. Bom_movies


```python
#previewing the top
bom_movies.head()
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
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415000000.0</td>
      <td>652000000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334200000.0</td>
      <td>691300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296000000.0</td>
      <td>664300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292600000.0</td>
      <td>535700000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238700000.0</td>
      <td>513900000</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the bottom 
bom_movies.tail()
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
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3382</th>
      <td>The Quake</td>
      <td>Magn.</td>
      <td>6200.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3383</th>
      <td>Edward II (2018 re-release)</td>
      <td>FM</td>
      <td>4800.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3384</th>
      <td>El Pacto</td>
      <td>Sony</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3385</th>
      <td>The Swan</td>
      <td>Synergetic</td>
      <td>2400.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3386</th>
      <td>An Actor Prepares</td>
      <td>Grav.</td>
      <td>1700.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
#determining the number of rows and columns
bom_movies.shape
```




    (3387, 5)




```python
#Checking the datatypes
bom_movies.dtypes
```




    title              object
    studio             object
    domestic_gross    float64
    foreign_gross      object
    year                int64
    dtype: object




```python
#previewing bom_movies information
bom_movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3387 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3387 non-null   object 
     1   studio          3382 non-null   object 
     2   domestic_gross  3359 non-null   float64
     3   foreign_gross   2037 non-null   object 
     4   year            3387 non-null   int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 132.4+ KB
    


```python
#previewing the summary statistics of bom_movies
bom_movies.describe()
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
      <th>domestic_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.359000e+03</td>
      <td>3387.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.874585e+07</td>
      <td>2013.958075</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.698250e+07</td>
      <td>2.478141</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+02</td>
      <td>2010.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.200000e+05</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000e+06</td>
      <td>2014.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.790000e+07</td>
      <td>2016.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.367000e+08</td>
      <td>2018.000000</td>
    </tr>
  </tbody>
</table>
</div>



Observations:

1.The mean domestic gross is about 28.7 million dollars, with a large standard deviation of about 67 million dollars. The minimum domestic gross is 100 dollars and the maximum is about 936.7 million dollars.

2.The DataFrame has 3387 rows and 5 columns.

3.The info method shows that there are some missing values in the studio, domestic_gross, and foreign_gross columns

4.The foreign_gross column is of the object data type, which suggests that it may contain non-numeric values.

#### b. Rt_movies


```python
#previewing the top of rt_movies
rt_movies.head()
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
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Drama|Science Fiction and Fantasy</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>Illeana Douglas delivers a superb performance ...</td>
      <td>R</td>
      <td>Drama|Musical and Performing Arts</td>
      <td>Allison Anders</td>
      <td>Allison Anders</td>
      <td>Sep 13, 1996</td>
      <td>Apr 18, 2000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>116 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>Michael Douglas runs afoul of a treacherous su...</td>
      <td>R</td>
      <td>Drama|Mystery and Suspense</td>
      <td>Barry Levinson</td>
      <td>Paul Attanasio|Michael Crichton</td>
      <td>Dec 9, 1994</td>
      <td>Aug 27, 1997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>128 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NR</td>
      <td>Drama|Romance</td>
      <td>Rodney Bennett</td>
      <td>Giles Cooper</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200 minutes</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the bottom of rt_movies
rt_movies.tail()
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
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1555</th>
      <td>1996</td>
      <td>Forget terrorists or hijackers -- there's a ha...</td>
      <td>R</td>
      <td>Action and Adventure|Horror|Mystery and Suspense</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Aug 18, 2006</td>
      <td>Jan 2, 2007</td>
      <td>$</td>
      <td>33,886,034</td>
      <td>106 minutes</td>
      <td>New Line Cinema</td>
    </tr>
    <tr>
      <th>1556</th>
      <td>1997</td>
      <td>The popular Saturday Night Live sketch was exp...</td>
      <td>PG</td>
      <td>Comedy|Science Fiction and Fantasy</td>
      <td>Steve Barron</td>
      <td>Terry Turner|Tom Davis|Dan Aykroyd|Bonnie Turner</td>
      <td>Jul 23, 1993</td>
      <td>Apr 17, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88 minutes</td>
      <td>Paramount Vantage</td>
    </tr>
    <tr>
      <th>1557</th>
      <td>1998</td>
      <td>Based on a novel by Richard Powell, when the l...</td>
      <td>G</td>
      <td>Classics|Comedy|Drama|Musical and Performing Arts</td>
      <td>Gordon Douglas</td>
      <td>NaN</td>
      <td>Jan 1, 1962</td>
      <td>May 11, 2004</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>111 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1558</th>
      <td>1999</td>
      <td>The Sandlot is a coming-of-age story about a g...</td>
      <td>PG</td>
      <td>Comedy|Drama|Kids and Family|Sports and Fitness</td>
      <td>David Mickey Evans</td>
      <td>David Mickey Evans|Robert Gunter</td>
      <td>Apr 1, 1993</td>
      <td>Jan 29, 2002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>101 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1559</th>
      <td>2000</td>
      <td>Suspended from the force, Paris cop Hubert is ...</td>
      <td>R</td>
      <td>Action and Adventure|Art House and Internation...</td>
      <td>NaN</td>
      <td>Luc Besson</td>
      <td>Sep 27, 2001</td>
      <td>Feb 11, 2003</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94 minutes</td>
      <td>Columbia Pictures</td>
    </tr>
  </tbody>
</table>
</div>




```python
#determining the number of rows and columns
rt_movies.shape
```




    (1560, 12)




```python
#previewing information on rt_movies
rt_movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1560 entries, 0 to 1559
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   id            1560 non-null   int64 
     1   synopsis      1498 non-null   object
     2   rating        1557 non-null   object
     3   genre         1552 non-null   object
     4   director      1361 non-null   object
     5   writer        1111 non-null   object
     6   theater_date  1201 non-null   object
     7   dvd_date      1201 non-null   object
     8   currency      340 non-null    object
     9   box_office    340 non-null    object
     10  runtime       1530 non-null   object
     11  studio        494 non-null    object
    dtypes: int64(1), object(11)
    memory usage: 146.4+ KB
    

Observations:

1. The currency, box_office, and studio columns have the most missing values, with only 340, 340, and 494 non-null entries respectively.
2. Values in genre column is separated by a vertical bar ( | ).
3. Runtime column is in string format I will drop the word "minutes" in the runtime column because it is in string format for easier analysis.

#### c. Rt_reviews


```python
#previewing the top of rt_reviews
rt_reviews.head()
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
      <th>review</th>
      <th>rating</th>
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>A distinctly gallows take on contemporary fina...</td>
      <td>3/5</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>It's an allegory in search of a meaning that n...</td>
      <td>NaN</td>
      <td>rotten</td>
      <td>Annalee Newitz</td>
      <td>0</td>
      <td>io9.com</td>
      <td>May 23, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>... life lived in a bubble in financial dealin...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>Sean Axmaker</td>
      <td>0</td>
      <td>Stream on Demand</td>
      <td>January 4, 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Continuing along a line introduced in last yea...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>Daniel Kasman</td>
      <td>0</td>
      <td>MUBI</td>
      <td>November 16, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>... a perverse twist on neorealism...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>NaN</td>
      <td>0</td>
      <td>Cinema Scope</td>
      <td>October 12, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the bottom of rt_reviews
rt_reviews.tail()
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
      <th>review</th>
      <th>rating</th>
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54427</th>
      <td>2000</td>
      <td>The real charm of this trifle is the deadpan c...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>Laura Sinagra</td>
      <td>1</td>
      <td>Village Voice</td>
      <td>September 24, 2002</td>
    </tr>
    <tr>
      <th>54428</th>
      <td>2000</td>
      <td>NaN</td>
      <td>1/5</td>
      <td>rotten</td>
      <td>Michael Szymanski</td>
      <td>0</td>
      <td>Zap2it.com</td>
      <td>September 21, 2005</td>
    </tr>
    <tr>
      <th>54429</th>
      <td>2000</td>
      <td>NaN</td>
      <td>2/5</td>
      <td>rotten</td>
      <td>Emanuel Levy</td>
      <td>0</td>
      <td>EmanuelLevy.Com</td>
      <td>July 17, 2005</td>
    </tr>
    <tr>
      <th>54430</th>
      <td>2000</td>
      <td>NaN</td>
      <td>2.5/5</td>
      <td>rotten</td>
      <td>Christopher Null</td>
      <td>0</td>
      <td>Filmcritic.com</td>
      <td>September 7, 2003</td>
    </tr>
    <tr>
      <th>54431</th>
      <td>2000</td>
      <td>NaN</td>
      <td>3/5</td>
      <td>fresh</td>
      <td>Nicolas Lacroix</td>
      <td>0</td>
      <td>Showbizz.net</td>
      <td>November 12, 2002</td>
    </tr>
  </tbody>
</table>
</div>




```python
#determining the number of rows and columns
rt_reviews.shape
```




    (54432, 8)




```python
#previewing information on rt_reviews
rt_reviews.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 54432 entries, 0 to 54431
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   id          54432 non-null  int64 
     1   review      48869 non-null  object
     2   rating      40915 non-null  object
     3   fresh       54432 non-null  object
     4   critic      51710 non-null  object
     5   top_critic  54432 non-null  int64 
     6   publisher   54123 non-null  object
     7   date        54432 non-null  object
    dtypes: int64(2), object(6)
    memory usage: 3.3+ MB
    


```python
#previewing the summary statistics
rt_reviews.describe()
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
      <th>top_critic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>54432.000000</td>
      <td>54432.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1045.706882</td>
      <td>0.240594</td>
    </tr>
    <tr>
      <th>std</th>
      <td>586.657046</td>
      <td>0.427448</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>542.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1083.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1541.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2000.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Observations:

1. The reviews seem to be sentences so they will not be of much use.
2. rt_movies and rt_reviews can be merged since they contain similar information.
3. The columns review,rating and critic appear have missing values  

#### d. Movie_budgets1


```python
#previewing the top movie_budgets

movie_budgets1.head()
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
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>May 20, 2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>May 1, 2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Dec 15, 2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
  </tbody>
</table>
</div>




```python
#determining the number of rows and columns
movie_budgets1.shape
```




    (5782, 6)




```python
#previewing information on movie_budgets
movie_budgets1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5782 entries, 0 to 5781
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   id                 5782 non-null   int64 
     1   release_date       5782 non-null   object
     2   movie              5782 non-null   object
     3   production_budget  5782 non-null   object
     4   domestic_gross     5782 non-null   object
     5   worldwide_gross    5782 non-null   object
    dtypes: int64(1), object(5)
    memory usage: 271.2+ KB
    

Observations:
1. The data has 2 types : integer and object.
2. The columns includes id, rerlease_date, production_budget, domestic_gross and worldwide_gross

 I will join the movie_budgets table with the rt_movies


```python
movie_budgets = pd.merge(movie_budgets1, rt_movies, on='id')
movie_budgets
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
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>May 29, 2009</td>
      <td>Up</td>
      <td>$175,000,000</td>
      <td>$293,004,164</td>
      <td>$731,463,377</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Mar 7, 2014</td>
      <td>Mr. Peabody &amp; Sherman</td>
      <td>$145,000,000</td>
      <td>$111,506,430</td>
      <td>$269,806,430</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Dec 17, 2010</td>
      <td>How Do You Know?</td>
      <td>$120,000,000</td>
      <td>$30,212,620</td>
      <td>$49,628,177</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Dec 11, 2015</td>
      <td>In the Heart of the Sea</td>
      <td>$100,000,000</td>
      <td>$25,020,758</td>
      <td>$89,693,309</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action and Adventure|Classics|Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>4677</th>
      <td>100</td>
      <td>Dec 31, 2013</td>
      <td>Heli</td>
      <td>$1,000,000</td>
      <td>$0</td>
      <td>$552,614</td>
      <td>Four old college friends in their forties come...</td>
      <td>NR</td>
      <td>Comedy|Drama</td>
      <td>Willem van de Sande Bakhuyzen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4678</th>
      <td>100</td>
      <td>Oct 11, 2013</td>
      <td>Escape from Tomorrow</td>
      <td>$650,000</td>
      <td>$171,962</td>
      <td>$171,962</td>
      <td>Four old college friends in their forties come...</td>
      <td>NR</td>
      <td>Comedy|Drama</td>
      <td>Willem van de Sande Bakhuyzen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4679</th>
      <td>100</td>
      <td>Jul 11, 2014</td>
      <td>As It Is in Heaven</td>
      <td>$450,000</td>
      <td>$13,486</td>
      <td>$13,486</td>
      <td>Four old college friends in their forties come...</td>
      <td>NR</td>
      <td>Comedy|Drama</td>
      <td>Willem van de Sande Bakhuyzen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4680</th>
      <td>100</td>
      <td>Dec 31, 2014</td>
      <td>Horse Camp</td>
      <td>$250,000</td>
      <td>$0</td>
      <td>$0</td>
      <td>Four old college friends in their forties come...</td>
      <td>NR</td>
      <td>Comedy|Drama</td>
      <td>Willem van de Sande Bakhuyzen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4681</th>
      <td>100</td>
      <td>Aug 30, 1972</td>
      <td>The Last House on the Left</td>
      <td>$87,000</td>
      <td>$3,100,000</td>
      <td>$3,100,000</td>
      <td>Four old college friends in their forties come...</td>
      <td>NR</td>
      <td>Comedy|Drama</td>
      <td>Willem van de Sande Bakhuyzen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>108 minutes</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>4682 rows × 17 columns</p>
</div>



#### e. IMBD Movies


```python
#viewing the list of tables in the imdb database
imdb_tables = pd.read_sql("""
SELECT name TableNames
FROM sqlite_master
WHERE type = 'table'
;
""",conn)
imdb_tables
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
      <th>TableNames</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>movie_basics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>directors</td>
    </tr>
    <tr>
      <th>2</th>
      <td>known_for</td>
    </tr>
    <tr>
      <th>3</th>
      <td>movie_akas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>movie_ratings</td>
    </tr>
    <tr>
      <th>5</th>
      <td>persons</td>
    </tr>
    <tr>
      <th>6</th>
      <td>principals</td>
    </tr>
    <tr>
      <th>7</th>
      <td>writers</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the movie_basics table
imovie_basics = pd.read_sql_query('SELECT * FROM movie_basics', conn)
imovie_basics.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography,Drama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.0</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0069204</td>
      <td>Sabse Bada Sukh</td>
      <td>Sabse Bada Sukh</td>
      <td>2018</td>
      <td>NaN</td>
      <td>Comedy,Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80.0</td>
      <td>Comedy,Drama,Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the persons table
ipersons = pd.read_sql_query('SELECT * FROM persons', conn)
ipersons.head()
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
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nm0061671</td>
      <td>Mary Ellen Bauder</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>miscellaneous,production_manager,producer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nm0061865</td>
      <td>Joseph Bauer</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>composer,music_department,sound_department</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nm0062070</td>
      <td>Bruce Baum</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>miscellaneous,actor,writer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nm0062195</td>
      <td>Axel Baumann</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>camera_department,cinematographer,art_department</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nm0062798</td>
      <td>Pete Baxter</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>production_designer,art_department,set_decorator</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the movie_ratings table
imovie_ratings = pd.read_sql_query('SELECT * FROM movie_ratings', conn)
imovie_ratings.head()
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
      <th>movie_id</th>
      <th>averagerating</th>
      <th>numvotes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt10356526</td>
      <td>8.3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt10384606</td>
      <td>8.9</td>
      <td>559</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt1042974</td>
      <td>6.4</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt1043726</td>
      <td>4.2</td>
      <td>50352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt1060240</td>
      <td>6.5</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the directors table
idirectors = pd.read_sql_query('SELECT * FROM directors', conn)
idirectors.head()
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
      <th>movie_id</th>
      <th>person_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0285252</td>
      <td>nm0899854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0462036</td>
      <td>nm1940585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0835418</td>
      <td>nm0151540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0835418</td>
      <td>nm0151540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0878654</td>
      <td>nm0089502</td>
    </tr>
  </tbody>
</table>
</div>



### I will join movie_ratings, directors and persons tables to movie_basics table since they all share movie_id and person_id.


```python
imdb = pd.read_sql("""
SELECT *
FROM movie_basics b
JOIN movie_ratings r
USING (movie_id)
JOIN directors d
USING (movie_id)
JOIN persons p
USING (person_id)
;
""",conn)
imdb.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography,Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>1944.0</td>
      <td>2011.0</td>
      <td>director,writer,actor</td>
    </tr>
  </tbody>
</table>
</div>




```python
#closing the connection to the database
conn.close()
```


```python
#previewing the top of the imdb dataframe
imdb.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>1921.0</td>
      <td>2004.0</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography,Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>1944.0</td>
      <td>2011.0</td>
      <td>director,writer,actor</td>
    </tr>
  </tbody>
</table>
</div>




```python
#previewing the bottom of the imdb dataframe
imdb.tail()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>birth_year</th>
      <th>death_year</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>181382</th>
      <td>tt9914642</td>
      <td>Albatross</td>
      <td>Albatross</td>
      <td>2017</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>8.5</td>
      <td>8</td>
      <td>nm5300859</td>
      <td>Chris Jordan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director,writer,editor</td>
    </tr>
    <tr>
      <th>181383</th>
      <td>tt9914642</td>
      <td>Albatross</td>
      <td>Albatross</td>
      <td>2017</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>8.5</td>
      <td>8</td>
      <td>nm5300859</td>
      <td>Chris Jordan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director,writer,editor</td>
    </tr>
    <tr>
      <th>181384</th>
      <td>tt9914942</td>
      <td>La vida sense la Sara Amat</td>
      <td>La vida sense la Sara Amat</td>
      <td>2019</td>
      <td>NaN</td>
      <td>None</td>
      <td>6.6</td>
      <td>5</td>
      <td>nm1716653</td>
      <td>Laura Jou</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>miscellaneous,actress,director</td>
    </tr>
    <tr>
      <th>181385</th>
      <td>tt9914942</td>
      <td>La vida sense la Sara Amat</td>
      <td>La vida sense la Sara Amat</td>
      <td>2019</td>
      <td>NaN</td>
      <td>None</td>
      <td>6.6</td>
      <td>5</td>
      <td>nm1716653</td>
      <td>Laura Jou</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>miscellaneous,actress,director</td>
    </tr>
    <tr>
      <th>181386</th>
      <td>tt9916160</td>
      <td>Drømmeland</td>
      <td>Drømmeland</td>
      <td>2019</td>
      <td>72.0</td>
      <td>Documentary</td>
      <td>6.5</td>
      <td>11</td>
      <td>nm5684093</td>
      <td>Joost van der Wiel</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>director,cinematographer,writer</td>
    </tr>
  </tbody>
</table>
</div>




```python
#determining the number of rows and columns
imdb.shape
```




    (181387, 13)




```python
#checking the information
imdb.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 181387 entries, 0 to 181386
    Data columns (total 13 columns):
     #   Column              Non-Null Count   Dtype  
    ---  ------              --------------   -----  
     0   movie_id            181387 non-null  object 
     1   primary_title       181387 non-null  object 
     2   original_title      181387 non-null  object 
     3   start_year          181387 non-null  int64  
     4   runtime_minutes     163584 non-null  float64
     5   genres              180047 non-null  object 
     6   averagerating       181387 non-null  float64
     7   numvotes            181387 non-null  int64  
     8   person_id           181387 non-null  object 
     9   primary_name        181387 non-null  object 
     10  birth_year          54805 non-null   float64
     11  death_year          1342 non-null    float64
     12  primary_profession  181262 non-null  object 
    dtypes: float64(4), int64(2), object(7)
    memory usage: 18.0+ MB
    


```python
#previewing the summary statistics of imdb 
imdb.describe()
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
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>birth_year</th>
      <th>death_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>181387.000000</td>
      <td>163584.000000</td>
      <td>181387.000000</td>
      <td>1.813870e+05</td>
      <td>54805.000000</td>
      <td>1342.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.309802</td>
      <td>97.789484</td>
      <td>6.217683</td>
      <td>4.955524e+03</td>
      <td>1969.097856</td>
      <td>2014.908346</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.536111</td>
      <td>194.434689</td>
      <td>1.388026</td>
      <td>3.760931e+04</td>
      <td>12.499740</td>
      <td>4.866581</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2010.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>5.000000e+00</td>
      <td>1870.000000</td>
      <td>1944.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2012.000000</td>
      <td>84.000000</td>
      <td>5.400000</td>
      <td>1.900000e+01</td>
      <td>1962.000000</td>
      <td>2014.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.000000</td>
      <td>94.000000</td>
      <td>6.300000</td>
      <td>6.600000e+01</td>
      <td>1971.000000</td>
      <td>2016.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>107.000000</td>
      <td>7.200000</td>
      <td>3.110000e+02</td>
      <td>1978.000000</td>
      <td>2018.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019.000000</td>
      <td>51420.000000</td>
      <td>10.000000</td>
      <td>1.841066e+06</td>
      <td>2004.000000</td>
      <td>2019.000000</td>
    </tr>
  </tbody>
</table>
</div>



Observations:
1. The genres are separated by commas.
2. There are columns with missing data: runtime_minutes, genres, birth_year, death_year and primary_profession.
3. The longest movie is 51,420 minutes and the lowest 3 minutes long and the standard deviation is at 194.4 from runtime_minutes.
4. The highest number of votes cast was 1,841,066 and the lowest 5 only.

# 7. Data Cleaning

This section prepares the data for exploratory analysis. It handles the problems with the datasets that have been observed while loading and exploring them. It will cover:
   - Duplicated rows
   - Missing values
   - Unwanted observations
   - Structural issues
   - Outliers
   - Wrong data types
   - Merges
   


### 7.1. Duplicate Rows


```python
#Defining a function for easier cleaning
def duplicates(df):
    duplicated_rows = df.duplicated()
    num_duplicated_rows = duplicated_rows.sum()
    print(f'Number of duplicated rows: {num_duplicated_rows}')

    
```


```python
#checking for duplicates in imdb dataframe
duplicates(imdb)
```

    Number of duplicated rows: 95357
    


```python
#checking for duplicates in bom_movies dataframe
duplicates(bom_movies)
```

    Number of duplicated rows: 0
    


```python
#checking for duplicates in rt_reviews dataframe
duplicates(rt_reviews)
```

    Number of duplicated rows: 9
    


```python
#checking for duplicates in rt_movies dataframe
duplicates(rt_movies)
```

    Number of duplicated rows: 0
    


```python
#checking for duplicates in movie_budgets dataframe
duplicates(movie_budgets)
```

    Number of duplicated rows: 0
    


```python
#dropping duplicates in imdb and rt_reviews
imdb.drop_duplicates(inplace = True)
rt_reviews.drop_duplicates(inplace = True)
```


```python
#confirming that duplicates have been dropped 
duplicates(imdb)
duplicates(rt_reviews)
```

    Number of duplicated rows: 0
    Number of duplicated rows: 0
    

### 7.2. Missing Values


```python
#defining a function that prints the percentage of null rows in all columns
def print_null_percentages(df):
    null_percentages = df.isnull().mean() * 100
    print('Percentage of null values in each column:')
    print(null_percentages)
```

**imdb**



```python
#finding percentage of nulls in imdb
print_null_percentages(imdb)
```

    Percentage of null values in each column:
    movie_id               0.000000
    primary_title          0.000000
    original_title         0.000000
    start_year             0.000000
    runtime_minutes       10.366151
    genres                 0.927583
    averagerating          0.000000
    numvotes               0.000000
    person_id              0.000000
    primary_name           0.000000
    birth_year            71.293735
    death_year            99.187493
    primary_profession     0.103452
    dtype: float64
    

- I will drop the columns birth_year and death_year since it does not provide relevant information.
- The other columns, 'runtime_minutes', 'genres' and 'primary_profession', with missing values will be dropped.


```python
#dropping birth_year and death_year
imdb.drop(['birth_year', 'death_year'], axis=1, inplace=True)
imdb.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography,Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.0</td>
      <td>Drama</td>
      <td>6.9</td>
      <td>4517</td>
      <td>nm0000080</td>
      <td>Orson Welles</td>
      <td>actor,director,writer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tt0069204</td>
      <td>Sabse Bada Sukh</td>
      <td>Sabse Bada Sukh</td>
      <td>2018</td>
      <td>NaN</td>
      <td>Comedy,Drama</td>
      <td>6.1</td>
      <td>13</td>
      <td>nm0611531</td>
      <td>Hrishikesh Mukherjee</td>
      <td>director,editor,writer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80.0</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.5</td>
      <td>119</td>
      <td>nm0749914</td>
      <td>Raoul Ruiz</td>
      <td>director,writer,producer</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dropping rows with missing values
imdb.dropna(subset = ['runtime_minutes','genres','primary_profession'],inplace = True)
```


```python
#confirming there are no missing values
imdb.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 76495 entries, 0 to 181386
    Data columns (total 11 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   movie_id            76495 non-null  object 
     1   primary_title       76495 non-null  object 
     2   original_title      76495 non-null  object 
     3   start_year          76495 non-null  int64  
     4   runtime_minutes     76495 non-null  float64
     5   genres              76495 non-null  object 
     6   averagerating       76495 non-null  float64
     7   numvotes            76495 non-null  int64  
     8   person_id           76495 non-null  object 
     9   primary_name        76495 non-null  object 
     10  primary_profession  76495 non-null  object 
    dtypes: float64(2), int64(2), object(7)
    memory usage: 7.0+ MB
    


```python
imdb.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action,Crime,Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography,Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.0</td>
      <td>Drama</td>
      <td>6.9</td>
      <td>4517</td>
      <td>nm0000080</td>
      <td>Orson Welles</td>
      <td>actor,director,writer</td>
    </tr>
    <tr>
      <th>8</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80.0</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.5</td>
      <td>119</td>
      <td>nm0749914</td>
      <td>Raoul Ruiz</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tt0100275</td>
      <td>The Wandering Soap Opera</td>
      <td>La Telenovela Errante</td>
      <td>2017</td>
      <td>80.0</td>
      <td>Comedy,Drama,Fantasy</td>
      <td>6.5</td>
      <td>119</td>
      <td>nm0765384</td>
      <td>Valeria Sarmiento</td>
      <td>editor,director,writer</td>
    </tr>
  </tbody>
</table>
</div>



**movie_budgets**


```python
#finding percentage of nulls in movie_budgets
print_null_percentages(movie_budgets)
```

    Percentage of null values in each column:
    id                    0.000000
    release_date          0.000000
    movie                 0.000000
    production_budget     0.000000
    domestic_gross        0.000000
    worldwide_gross       0.000000
    synopsis              3.716361
    rating                1.238787
    genre                 1.238787
    director             17.257582
    writer               27.125160
    theater_date         29.538659
    dvd_date             29.538659
    currency             76.527125
    box_office           76.527125
    runtime               3.716361
    studio               66.616830
    dtype: float64
    

- The columns rating, genre, runtime have a small number of null values. These will be handled by dropping the associated rows.
- Columns synopsis, director, writer, theater_date, dvd_date and currency are to be dropped as they are not required for this analysis.
- I will also drop box_office since worldwide_gross can also cover that.


```python
#dropping unnecessary columns 
movie_budgets.drop(['synopsis','director','writer','theater_date','dvd_date','currency','box_office', 'studio'],
               axis = 1,inplace=True)
```


```python
#dropping rows with null values
movie_budgets.dropna(subset = ['rating','genre','runtime'],inplace = True)
```


```python
#confirming there are no missing values
movie_budgets.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4508 entries, 0 to 4681
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   id                 4508 non-null   int64 
     1   release_date       4508 non-null   object
     2   movie              4508 non-null   object
     3   production_budget  4508 non-null   object
     4   domestic_gross     4508 non-null   object
     5   worldwide_gross    4508 non-null   object
     6   rating             4508 non-null   object
     7   genre              4508 non-null   object
     8   runtime            4508 non-null   object
    dtypes: int64(1), object(8)
    memory usage: 352.2+ KB
    

**bom_movies**


```python
print_null_percentages(bom_movies)
```

    Percentage of null values in each column:
    title              0.000000
    studio             0.147623
    domestic_gross     0.826690
    foreign_gross     39.858282
    year               0.000000
    dtype: float64
    

- Since the foreign_gross 40% null and it is in object format I will replace it with 0 so that it doesnt have much effect.
- The studio and the domestic_gross columns only have 0.827% and 0.148% missing values respectively. Hence I will drop the rows.


```python
#dropping null rows
bom_movies.dropna(subset = ['studio', 'domestic_gross'], inplace=True)
```


```python
#replacing missing values in foreign_gross column with 0
bom_movies.foreign_gross.fillna('0',inplace = True)
```


```python
bom_movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3356 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3356 non-null   object 
     1   studio          3356 non-null   object 
     2   domestic_gross  3356 non-null   float64
     3   foreign_gross   3356 non-null   object 
     4   year            3356 non-null   int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 157.3+ KB
    

**rt_reviews**


```python
print_null_percentages(rt_reviews)
```

    Percentage of null values in each column:
    id             0.000000
    review        10.208919
    rating        24.835088
    fresh          0.000000
    critic         4.985025
    top_critic     0.000000
    publisher      0.567775
    date           0.000000
    dtype: float64
    


```python
rt_reviews.head()
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
      <th>review</th>
      <th>rating</th>
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>A distinctly gallows take on contemporary fina...</td>
      <td>3/5</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>It's an allegory in search of a meaning that n...</td>
      <td>NaN</td>
      <td>rotten</td>
      <td>Annalee Newitz</td>
      <td>0</td>
      <td>io9.com</td>
      <td>May 23, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>... life lived in a bubble in financial dealin...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>Sean Axmaker</td>
      <td>0</td>
      <td>Stream on Demand</td>
      <td>January 4, 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Continuing along a line introduced in last yea...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>Daniel Kasman</td>
      <td>0</td>
      <td>MUBI</td>
      <td>November 16, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>... a perverse twist on neorealism...</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>NaN</td>
      <td>0</td>
      <td>Cinema Scope</td>
      <td>October 12, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
rt_reviews.rating.unique()
```




    array(['3/5', nan, 'C', '2/5', 'B-', '2/4', 'B', '3/4', '4/5', '4/4',
           '6/10', '1/4', '8', '2.5/4', '4/10', '2.0/5', '3/10', '7/10', 'A-',
           '5/5', 'F', '3.5/4', 'D+', '1.5/4', '3.5/5', '8/10', 'B+', '9/10',
           '2.5/5', '7.5/10', '5.5/10', 'C-', '1.5/5', '1/5', '5/10', 'C+',
           '0/5', '6', '0.5/4', 'D', '3.1/5', '3/6', '4.5/5', '0/4', '2/10',
           'D-', '7', '1/10', '3', 'A+', 'A', '4.0/4', '9.5/10', '2.5',
           '2.1/2', '6.5/10', '3.7/5', '8.4/10', '9', '1', '7.2/10', '2.2/5',
           '0.5/10', '5', '0', '2', '4.5', '7.7', '5.0/5', '8.5/10', '3.0/5',
           '0.5/5', '1.5/10', '3.0/4', '2.3/10', '4.5/10', '4/6', '3.5',
           '8.6/10', '6/8', '2.0/4', '2.7', '4.2/10', '5.8', '4', '7.1/10',
           '5/4', 'N', '3.5/10', '5.8/10', 'R', '4.0/5', '0/10', '5.0/10',
           '5.9/10', '2.4/5', '1.9/5', '4.9', '7.4/10', '1.5', '2.3/4',
           '8.8/10', '4.0/10', '2.2', '3.8/10', '6.8/10', '7.3', '7.0/10',
           '3.2', '4.2', '8.4', '5.5/5', '6.3/10', '7.6/10', '8.1/10',
           '3.6/5', '2/6', '7.7/10', '1.8', '8.9/10', '8.9', '8.2/10',
           '8.3/10', '2.6/6', '4.1/10', '2.5/10', 'F+', '6.0/10', '1.0/4',
           '7.9/10', '8.7/10', '4.3/10', '9.6/10', '9.0/10', '4.0', '1.7',
           '7.9', '6.7', '8.0/10', '9.2/10', '5.2', '5.9', '3.7', '4.7',
           '6.2/10', '1/6', '8.2', '2.6/5', '3.4', '9.7', '3.3/5', '3.8/5',
           '1/2', '7.4', '4.8', '1.6/5', '2/2', '1-5', '1.0', '4.3/5', '5/6',
           '9.2', '2.7/5', '4.9/10', '3.0', '3.1', '7.8/10', 'F-', '2.3/5',
           '3.0/10', '3/2', '7.8', '4.2/5', '9.0', '7.3/10', '4.4/5',
           '6.9/10', '0/6', 'T', '6.2', '3.3', '9.8', '8.5', '1.0/5', '4.1',
           '7.1', '3 1/2'], dtype=object)



- I will drop the review column as I cannot compare all the different opinions. 
- I will drop the rows in the critic and publisher columns.
- I will also drop the rating column since it has too many incosistencies.


```python
#dropping review and rating columns
rt_reviews.drop(['rating', 'review'], axis=1,inplace=True)
```


```python
#dropping rows with null values
rt_reviews.dropna(inplace=True)
```


```python
#confirming there are no missing values
rt_reviews.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 51433 entries, 0 to 54431
    Data columns (total 6 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   id          51433 non-null  int64 
     1   fresh       51433 non-null  object
     2   critic      51433 non-null  object
     3   top_critic  51433 non-null  int64 
     4   publisher   51433 non-null  object
     5   date        51433 non-null  object
    dtypes: int64(2), object(4)
    memory usage: 2.7+ MB
    

### 7.3 Structural Issues

The genre columns of imdb,rt_movies and movie_budgets1 have structural issues. They are separated by commas for imdb movies and vertical bars and 'and' for movie_budgets. I will be deal with this by splitting the columns on their specific delimiters using the `.split()` method then, using the `.explode()` method, transforming into separate rows and retaining all other column values. This will increase the number of rows.

**a. imdb movies**


```python
#imdb
imdb = imdb.assign(genres=imdb.genres.str.split(',')).explode('genres')
imdb.head()
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Crime</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
  </tbody>
</table>
</div>



**b. movie_budgets**


```python
#movie_budgets
movie_budgets = movie_budgets.assign(genre=movie_budgets.genre.str.split('|')).explode('genre')
movie_budgets = movie_budgets.assign(genre=movie_budgets.genre.str.split('and')).explode('genre')
movie_budgets.head()
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
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>rating</th>
      <th>genre</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Action</td>
      <td>104 minutes</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Adventure</td>
      <td>104 minutes</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Classics</td>
      <td>104 minutes</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Drama</td>
      <td>104 minutes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>May 29, 2009</td>
      <td>Up</td>
      <td>$175,000,000</td>
      <td>$293,004,164</td>
      <td>$731,463,377</td>
      <td>R</td>
      <td>Action</td>
      <td>104 minutes</td>
    </tr>
  </tbody>
</table>
</div>



**c. rt_movies**


```python
rt_movies = rt_movies.assign(genre=rt_movies.genre.str.split('|')).explode('genre')
rt_movies = rt_movies.assign(genre=rt_movies.genre.str.split('and')).explode('genre')
rt_movies.head()
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
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Action</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Adventure</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Classics</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>This gritty, fast-paced, and innovative police...</td>
      <td>R</td>
      <td>Drama</td>
      <td>William Friedkin</td>
      <td>Ernest Tidyman</td>
      <td>Oct 9, 1971</td>
      <td>Sep 25, 2001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104 minutes</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Drama</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
  </tbody>
</table>
</div>



**d. bom_movies**


```python
#Removing the commas from the foreign_gross column
bom_movies['foreign_gross'] = bom_movies['foreign_gross'].str.replace(',', '')
```

### 7.4. Outliers

As earlier seen, there imbd dataframe contains a movie that is 51420 minutes long and one that is 3. Movies over 200 minutes long and those under 40 minutes will be considered outliers so as not to skew the data so much.


```python
#Plotting boxplot to visualize the outliers
sns.boxplot(data = imdb, x = 'runtime_minutes', showfliers=True);
```




    <AxesSubplot:xlabel='runtime_minutes'>




    
![png](Movie_Analysis_files/Movie_Analysis_111_1.png)
    



```python
#Selecting the rows with the outliers
imdb.loc[(imdb.runtime_minutes > 200) | (imdb.runtime_minutes < 40)]
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>143</th>
      <td>tt0396123</td>
      <td>Den milde smerte</td>
      <td>Den milde smerte</td>
      <td>2010</td>
      <td>280.0</td>
      <td>Drama</td>
      <td>7.8</td>
      <td>6</td>
      <td>nm0104888</td>
      <td>Carsten Brandt</td>
      <td>actor,director,writer</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>tt10243660</td>
      <td>A Tale of Two Kitchens</td>
      <td>A Tale of Two Kitchens</td>
      <td>2019</td>
      <td>29.0</td>
      <td>Documentary</td>
      <td>6.7</td>
      <td>104</td>
      <td>nm1970269</td>
      <td>Trisha Ziff</td>
      <td>producer,director,writer</td>
    </tr>
    <tr>
      <th>1705</th>
      <td>tt10244756</td>
      <td>Ang hupa</td>
      <td>Ang hupa</td>
      <td>2019</td>
      <td>276.0</td>
      <td>Sci-Fi</td>
      <td>7.2</td>
      <td>5</td>
      <td>nm0225010</td>
      <td>Lav Diaz</td>
      <td>writer,director,editor</td>
    </tr>
    <tr>
      <th>2216</th>
      <td>tt1113829</td>
      <td>George Harrison: Living in the Material World</td>
      <td>George Harrison: Living in the Material World</td>
      <td>2011</td>
      <td>208.0</td>
      <td>Biography</td>
      <td>8.2</td>
      <td>9372</td>
      <td>nm0000217</td>
      <td>Martin Scorsese</td>
      <td>producer,director,actor</td>
    </tr>
    <tr>
      <th>2216</th>
      <td>tt1113829</td>
      <td>George Harrison: Living in the Material World</td>
      <td>George Harrison: Living in the Material World</td>
      <td>2011</td>
      <td>208.0</td>
      <td>Documentary</td>
      <td>8.2</td>
      <td>9372</td>
      <td>nm0000217</td>
      <td>Martin Scorsese</td>
      <td>producer,director,actor</td>
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
    </tr>
    <tr>
      <th>179676</th>
      <td>tt9318514</td>
      <td>Reason</td>
      <td>Vivek</td>
      <td>2018</td>
      <td>261.0</td>
      <td>Documentary</td>
      <td>9.0</td>
      <td>44</td>
      <td>nm0666674</td>
      <td>Anand Patwardhan</td>
      <td>director,editor,producer</td>
    </tr>
    <tr>
      <th>180302</th>
      <td>tt9573980</td>
      <td>Leaving Neverland</td>
      <td>Leaving Neverland</td>
      <td>2019</td>
      <td>240.0</td>
      <td>Documentary</td>
      <td>7.1</td>
      <td>19632</td>
      <td>nm0715371</td>
      <td>Dan Reed</td>
      <td>director,producer,writer</td>
    </tr>
    <tr>
      <th>180784</th>
      <td>tt9749570</td>
      <td>Heimat Is a Space in Time</td>
      <td>Heimat ist ein Raum aus Zeit</td>
      <td>2019</td>
      <td>218.0</td>
      <td>Documentary</td>
      <td>7.8</td>
      <td>14</td>
      <td>nm0374656</td>
      <td>Thomas Heise</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>181098</th>
      <td>tt9865446</td>
      <td>Siege</td>
      <td>Siege</td>
      <td>2019</td>
      <td>16.0</td>
      <td>Sci-Fi</td>
      <td>8.5</td>
      <td>32</td>
      <td>nm10490240</td>
      <td>Deeptanshu Sinha</td>
      <td>director,writer,art_department</td>
    </tr>
    <tr>
      <th>181098</th>
      <td>tt9865446</td>
      <td>Siege</td>
      <td>Siege</td>
      <td>2019</td>
      <td>16.0</td>
      <td>Thriller</td>
      <td>8.5</td>
      <td>32</td>
      <td>nm10490240</td>
      <td>Deeptanshu Sinha</td>
      <td>director,writer,art_department</td>
    </tr>
  </tbody>
</table>
<p>871 rows × 11 columns</p>
</div>




```python
#Dropping outliers
imdb = imdb.loc[(imdb.runtime_minutes>=40) & (imdb.runtime_minutes<=200)]
imdb.shape
```




    (137604, 11)




```python
#Previewing boxplot again to visualize the outliers
sns.boxplot(data = imdb, x = 'runtime_minutes');
```




    <AxesSubplot:xlabel='runtime_minutes'>




    
![png](Movie_Analysis_files/Movie_Analysis_114_1.png)
    


The boxplot shows that most movies are between 80 to 105 minutes long.

### 7.5 Wrong data types

**a. movie_budgets**

The word "minutes" is in the runtime column. I will remove it and convert to integers. 


```python
#Removing minutes and converting to integer
movie_budgets['runtime'] = movie_budgets['runtime'].str.replace('minutes', '').astype(int)

```


```python
movie_budgets['runtime']
```




    0       104
    0       104
    0       104
    0       104
    1       104
           ... 
    4679    108
    4680    108
    4680    108
    4681    108
    4681    108
    Name: runtime, Length: 13523, dtype: int32



**b. bom_movies**


I will convert the foreign_gross column from an object to a float data type for easier manipulation.


```python
#Converting the foreign_gross column to float
bom_movies['foreign_gross'] = bom_movies['foreign_gross'].astype(float)
```

### 7.6. Merges

a. Movie_budgets and rt_reviews can be merged using the column id.


```python
movie_budgets.head()
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
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>rating</th>
      <th>genre</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Action</td>
      <td>104</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Adventure</td>
      <td>104</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Classics</td>
      <td>104</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
      <td>R</td>
      <td>Drama</td>
      <td>104</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>May 29, 2009</td>
      <td>Up</td>
      <td>$175,000,000</td>
      <td>$293,004,164</td>
      <td>$731,463,377</td>
      <td>R</td>
      <td>Action</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>




```python
rt_reviews.head()
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
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>rotten</td>
      <td>Annalee Newitz</td>
      <td>0</td>
      <td>io9.com</td>
      <td>May 23, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>fresh</td>
      <td>Sean Axmaker</td>
      <td>0</td>
      <td>Stream on Demand</td>
      <td>January 4, 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>fresh</td>
      <td>Daniel Kasman</td>
      <td>0</td>
      <td>MUBI</td>
      <td>November 16, 2017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>fresh</td>
      <td>Michelle Orange</td>
      <td>0</td>
      <td>Capital New York</td>
      <td>September 11, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merging the two dataframes
movie_budgets_merged = rt_reviews.merge(movie_budgets, how='inner', on='id')
movie_budgets_merged

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
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>rating</th>
      <th>genre</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
      <td>R</td>
      <td>Drama</td>
      <td>108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
      <td>R</td>
      <td>Science Fiction</td>
      <td>108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
      <td>R</td>
      <td>Fantasy</td>
      <td>108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>Nov 21, 2018</td>
      <td>Ralph Breaks The Internet</td>
      <td>$175,000,000</td>
      <td>$201,091,711</td>
      <td>$524,283,695</td>
      <td>R</td>
      <td>Drama</td>
      <td>108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>Nov 21, 2018</td>
      <td>Ralph Breaks The Internet</td>
      <td>$175,000,000</td>
      <td>$201,091,711</td>
      <td>$524,283,695</td>
      <td>R</td>
      <td>Science Fiction</td>
      <td>108</td>
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
    </tr>
    <tr>
      <th>464694</th>
      <td>99</td>
      <td>rotten</td>
      <td>Robert Denerstein</td>
      <td>1</td>
      <td>Denver Rocky Mountain News</td>
      <td>December 23, 2002</td>
      <td>May 9, 2008</td>
      <td>Poultrygeist: Night of the Chicken Dead</td>
      <td>$450,000</td>
      <td>$13,804</td>
      <td>$22,623</td>
      <td>PG-13</td>
      <td>Romance</td>
      <td>102</td>
    </tr>
    <tr>
      <th>464695</th>
      <td>99</td>
      <td>rotten</td>
      <td>Robert Denerstein</td>
      <td>1</td>
      <td>Denver Rocky Mountain News</td>
      <td>December 23, 2002</td>
      <td>Feb 3, 2015</td>
      <td>UnDivided</td>
      <td>$250,000</td>
      <td>$0</td>
      <td>$0</td>
      <td>PG-13</td>
      <td>Comedy</td>
      <td>102</td>
    </tr>
    <tr>
      <th>464696</th>
      <td>99</td>
      <td>rotten</td>
      <td>Robert Denerstein</td>
      <td>1</td>
      <td>Denver Rocky Mountain News</td>
      <td>December 23, 2002</td>
      <td>Feb 3, 2015</td>
      <td>UnDivided</td>
      <td>$250,000</td>
      <td>$0</td>
      <td>$0</td>
      <td>PG-13</td>
      <td>Romance</td>
      <td>102</td>
    </tr>
    <tr>
      <th>464697</th>
      <td>99</td>
      <td>rotten</td>
      <td>Robert Denerstein</td>
      <td>1</td>
      <td>Denver Rocky Mountain News</td>
      <td>December 23, 2002</td>
      <td>Jul 7, 2015</td>
      <td>Tiger Orange</td>
      <td>$100,000</td>
      <td>$0</td>
      <td>$0</td>
      <td>PG-13</td>
      <td>Comedy</td>
      <td>102</td>
    </tr>
    <tr>
      <th>464698</th>
      <td>99</td>
      <td>rotten</td>
      <td>Robert Denerstein</td>
      <td>1</td>
      <td>Denver Rocky Mountain News</td>
      <td>December 23, 2002</td>
      <td>Jul 7, 2015</td>
      <td>Tiger Orange</td>
      <td>$100,000</td>
      <td>$0</td>
      <td>$0</td>
      <td>PG-13</td>
      <td>Romance</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
<p>464699 rows × 14 columns</p>
</div>



b .rt_movies and rt_reviews I will merge on the column id since they are from the same source.


```python
#merging the two dataframes
rt_merged = rt_reviews.merge(rt_movies, how = 'inner',on = 'id')
rt_merged.head()
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
      <th>fresh</th>
      <th>critic</th>
      <th>top_critic</th>
      <th>publisher</th>
      <th>date</th>
      <th>synopsis</th>
      <th>rating</th>
      <th>genre</th>
      <th>director</th>
      <th>writer</th>
      <th>theater_date</th>
      <th>dvd_date</th>
      <th>currency</th>
      <th>box_office</th>
      <th>runtime</th>
      <th>studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Drama</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Science Fiction</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>fresh</td>
      <td>PJ Nabarro</td>
      <td>0</td>
      <td>Patrick Nabarro</td>
      <td>November 10, 2018</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Fantasy</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>rotten</td>
      <td>Annalee Newitz</td>
      <td>0</td>
      <td>io9.com</td>
      <td>May 23, 2018</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Drama</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>rotten</td>
      <td>Annalee Newitz</td>
      <td>0</td>
      <td>io9.com</td>
      <td>May 23, 2018</td>
      <td>New York City, not-too-distant-future: Eric Pa...</td>
      <td>R</td>
      <td>Science Fiction</td>
      <td>David Cronenberg</td>
      <td>David Cronenberg|Don DeLillo</td>
      <td>Aug 17, 2012</td>
      <td>Jan 1, 2013</td>
      <td>$</td>
      <td>600,000</td>
      <td>108 minutes</td>
      <td>Entertainment One</td>
    </tr>
  </tbody>
</table>
</div>



# 8. Exploratory Data Analysis

This section will analyse the data and creating visualisations that answer the following business questions that will lead to appropriate recommendations:
1. What are the most popular genres?
2. What is the distribution of ratings among the movies?
3. Who are the best directors?
4. What is the relationship between domestic gross and worldwide gross?
5. Which are the most successful studios?

## 8.1. Most Popular Genres

**a. Using the IMDB Dataset**

First I will get the number of the most common genres available


```python
imdb.head(5)
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Action</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Crime</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>0</th>
      <td>tt0063540</td>
      <td>Sunghursh</td>
      <td>Sunghursh</td>
      <td>2013</td>
      <td>175.0</td>
      <td>Drama</td>
      <td>7.0</td>
      <td>77</td>
      <td>nm0712540</td>
      <td>Harnam Singh Rawail</td>
      <td>director,writer,producer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Biography</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tt0066787</td>
      <td>One Day Before the Rainy Season</td>
      <td>Ashad Ka Ek Din</td>
      <td>2019</td>
      <td>114.0</td>
      <td>Drama</td>
      <td>7.2</td>
      <td>43</td>
      <td>nm0002411</td>
      <td>Mani Kaul</td>
      <td>director,writer,actor</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Calculating frequency of each genre to determine which genres are the most popular.
genre_counts = imdb['genres'].value_counts()
genre_counts
```




    Drama          32009
    Documentary    19797
    Comedy         18188
    Horror          8813
    Thriller        8294
    Action          7124
    Romance         6564
    Crime           4814
    Biography       4348
    Adventure       4306
    Family          3644
    Mystery         3300
    History         3181
    Sci-Fi          2471
    Fantasy         2324
    Animation       2238
    Music           2237
    Sport           1332
    War              903
    Musical          710
    News             702
    Western          281
    Reality-TV        19
    Adult              3
    Game-Show          2
    Name: genres, dtype: int64




```python
#Setting the style
plt.style.use('ggplot')
```


```python
#Sorting the genre counts in ascending order
genre_counts_asc = genre_counts.sort_values(ascending=True)
```


```python
# Plotting a bar chart to visualize the top genre counts
genre_counts_asc.plot(kind='barh')
plt.title('IMDB Genre Availability')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()
fig = plt.figure()
fig.savefig('Images/IMDB Genre Availability.png');
```


    
![png](Movie_Analysis_files/Movie_Analysis_140_0.png)
    



    <Figure size 432x288 with 0 Axes>


- I will Group the genres by the ratings.
- The averagerating column will be categorised into "High","Average" and "Low" to help in grouping the data.
- I will add a cap of at least 1000 votes since some movies have high ratings because the number of votes are very few.


```python
#Calculating the highest value, lowest values and mean values in the averagerating column
max_rating = imdb['averagerating'].max()
min_rating = imdb['averagerating'].min()
mean_rating = imdb['averagerating'].mean()
print(f'Highest rating: {max_rating}')
print(f'Lowest rating: {min_rating}')
print(f'Mean rating: {mean_rating}')
```

    Highest rating: 10.0
    Lowest rating: 1.0
    Mean rating: 6.29719775587919
    


```python
#Defining a function to categorize the ratings
def categorize(averagerating):
    if averagerating > 8:
        return 'High'
    elif averagerating >=5 and averagerating <=7:
        return 'Average'
    else:
        return'Low'
    
#creating a new column that categorises the average rating into high, average and low
imdb['rating'] = imdb['averagerating'].apply(categorize)
imdb[['averagerating', 'rating']]  
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
      <th>averagerating</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>181377</th>
      <td>6.2</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>181378</th>
      <td>6.2</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>181380</th>
      <td>8.7</td>
      <td>High</td>
    </tr>
    <tr>
      <th>181380</th>
      <td>8.7</td>
      <td>High</td>
    </tr>
    <tr>
      <th>181386</th>
      <td>6.5</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
<p>137604 rows × 2 columns</p>
</div>




```python
max_votes = imdb['numvotes'].max()
min_votes = imdb['numvotes'].min()
mean_votes = imdb['numvotes'].mean()
print(f'Highest votes: {max_votes}')
print(f'Lowest votes: {min_votes}')
print(f'Mean votes: {mean_votes}')
```

    Highest votes: 1841066
    Lowest votes: 5
    Mean votes: 5534.7357998314
    


```python
#Creating a boolean mask to select rows where the value in the numvotes column is at more than 1000
mask = imdb['numvotes'] >= 1000
```


```python
#creating a new dataframe to house the number of votes are greater than 1000
imdbnew= imdb[mask]
imdbnew
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.0</td>
      <td>Drama</td>
      <td>6.9</td>
      <td>4517</td>
      <td>nm0000080</td>
      <td>Orson Welles</td>
      <td>actor,director,writer</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Action</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Animation</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Comedy</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>55</th>
      <td>tt0293069</td>
      <td>Dark Blood</td>
      <td>Dark Blood</td>
      <td>2012</td>
      <td>86.0</td>
      <td>Thriller</td>
      <td>6.6</td>
      <td>1053</td>
      <td>nm0806293</td>
      <td>George Sluizer</td>
      <td>director,producer,writer</td>
      <td>Average</td>
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
    </tr>
    <tr>
      <th>180277</th>
      <td>tt9558612</td>
      <td>PM Narendra Modi</td>
      <td>PM Narendra Modi</td>
      <td>2019</td>
      <td>136.0</td>
      <td>Drama</td>
      <td>3.7</td>
      <td>4057</td>
      <td>nm1293225</td>
      <td>Omung Kumar</td>
      <td>art_department,art_director,production_designer</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>180282</th>
      <td>tt9562694</td>
      <td>Alien Warfare</td>
      <td>Alien Warfare</td>
      <td>2019</td>
      <td>88.0</td>
      <td>Action</td>
      <td>2.6</td>
      <td>1509</td>
      <td>nm1371053</td>
      <td>Jeremiah Jones</td>
      <td>director,writer,editor</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>180282</th>
      <td>tt9562694</td>
      <td>Alien Warfare</td>
      <td>Alien Warfare</td>
      <td>2019</td>
      <td>88.0</td>
      <td>Sci-Fi</td>
      <td>2.6</td>
      <td>1509</td>
      <td>nm1371053</td>
      <td>Jeremiah Jones</td>
      <td>director,writer,editor</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>180818</th>
      <td>tt9778170</td>
      <td>Panodrama</td>
      <td>Panodrama</td>
      <td>2019</td>
      <td>64.0</td>
      <td>Documentary</td>
      <td>3.1</td>
      <td>2220</td>
      <td>nm2792327</td>
      <td>Tommy Robinson</td>
      <td>director,miscellaneous</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>180896</th>
      <td>tt9815714</td>
      <td>The Hard Way</td>
      <td>The Hard Way</td>
      <td>2019</td>
      <td>92.0</td>
      <td>Action</td>
      <td>4.7</td>
      <td>1214</td>
      <td>nm0915394</td>
      <td>Keoni Waxman</td>
      <td>producer,director,writer</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
<p>24280 rows × 12 columns</p>
</div>




```python
#groping imdb1 by genre and rating
genres = imdbnew.groupby(['genres', 'rating'])['movie_id'].count()
```


```python
#sorting the dataframe by high ratings
genres = genres.unstack().sort_values('High',ascending = False)[:10]
```


```python
#setting the plot style
plt.style.use('ggplot')
```


```python
#plotting a bar graph of most popular genres
fig,ax = plt.subplots(figsize=(15,10))

title = 'Top 10 Genres According to IMDB'
y_label = 'Number of High Ratings'
x_label = 'Genre'

genres.High.plot(kind = 'bar')
ax.set_title(title,fontsize=15)
ax.set_ylabel(y_label,fontsize=15)
ax.set_xlabel(x_label,fontsize=15)
plt.xticks(rotation = 60,fontsize=12)
fig.savefig('Images/IMDB Top 10 Genres.png');
```


    
![png](Movie_Analysis_files/Movie_Analysis_150_0.png)
    


b .Using Rotten Tomatoes rating system, fresh vs rotten from rt_merged


```python
#grouping rt_merged by genre and fresh (rating)
rt_genres = rt_merged.groupby(['genre','fresh']).count()['id'].unstack()
```


```python
#getting top 10 genres
rt_genres = rt_genres.sort_values('fresh',ascending = False)[:10]
rt_genres
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
      <th>fresh</th>
      <th>fresh</th>
      <th>rotten</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Drama</th>
      <td>20558</td>
      <td>11090</td>
    </tr>
    <tr>
      <th>Comedy</th>
      <td>12093</td>
      <td>8506</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>6909</td>
      <td>5293</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>6909</td>
      <td>5293</td>
    </tr>
    <tr>
      <th>Suspense</th>
      <td>6490</td>
      <td>4623</td>
    </tr>
    <tr>
      <th>Mystery</th>
      <td>6490</td>
      <td>4623</td>
    </tr>
    <tr>
      <th>Romance</th>
      <td>4856</td>
      <td>2782</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>3811</td>
      <td>3181</td>
    </tr>
    <tr>
      <th>Science Fiction</th>
      <td>3811</td>
      <td>3181</td>
    </tr>
    <tr>
      <th>International</th>
      <td>3179</td>
      <td>1022</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plotting a bar graph of most popular genres according to rotten tomatoes
fig,ax = plt.subplots(figsize=(15,10))

title = 'Rotten Tomatoes Top 10 Genres'
y_label = 'Number of Fresh Ratings'
x_label = 'Genre'

rt_genres.fresh.plot(kind = 'bar')
ax.set_title(title,fontsize=15)
ax.set_ylabel(y_label,fontsize=15)
ax.set_xlabel(x_label,fontsize=15)
plt.xticks(rotation = 60,fontsize=12);
fig.savefig('Images/Rotten Tomatoes Top 10 Genres.png')
```


    
![png](Movie_Analysis_files/Movie_Analysis_154_0.png)
    


Observations:
- Drama is the most popular genre followed by Documentary and Comedy  
- Action, Adventure and Suspense are also pretty popular

## 8.2 Distribution of ratings among the movies

Here, using rt_merged dataframe, we will find the film ratings for the most popular movies


```python
#grouping rt_merged by rating and fresh rating system
rt_ratings = rt_merged.groupby(['rating','fresh'])['id'].count().unstack()
```


```python
#sorting values by fresh rating
rt_ratings = rt_ratings.sort_values('fresh',ascending = False)
rt_ratings
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
      <th>fresh</th>
      <th>fresh</th>
      <th>rotten</th>
    </tr>
    <tr>
      <th>rating</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>R</th>
      <td>39568</td>
      <td>23009</td>
    </tr>
    <tr>
      <th>PG-13</th>
      <td>25764</td>
      <td>24147</td>
    </tr>
    <tr>
      <th>PG</th>
      <td>19352</td>
      <td>10377</td>
    </tr>
    <tr>
      <th>NR</th>
      <td>6164</td>
      <td>1845</td>
    </tr>
    <tr>
      <th>G</th>
      <td>3400</td>
      <td>1144</td>
    </tr>
  </tbody>
</table>
</div>




```python
#plotting a bar graph of the ratings
fig,ax = plt.subplots(figsize=(15,10))

title = 'Top Film Ratings'
y_label = 'Number of Movies'
x_label = 'Rating'

rt_ratings.fresh.plot(kind = 'bar')
ax.set_title(title,fontsize=15)
ax.set_ylabel(y_label,fontsize=15)
ax.set_xlabel(x_label,fontsize=15)
plt.xticks(rotation = 60,fontsize = 15)
fig.savefig('Images/Top Film Ratings.png');
```


    
![png](Movie_Analysis_files/Movie_Analysis_160_0.png)
    


Observations:
- R rated movies are the most popular, followed by PG-13 with G rated movies being the least popular.

## 8.3. The best directors

Here I will;
- Use the imdbnew dataframe as it is already sorted.
- Find the best directors for the most popular categories, drama, comedy and action.
- Use the rows where rating is 'High' and number of votes greater than 5000.


```python
imdbnew.head(5)
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
      <th>movie_id</th>
      <th>primary_title</th>
      <th>original_title</th>
      <th>start_year</th>
      <th>runtime_minutes</th>
      <th>genres</th>
      <th>averagerating</th>
      <th>numvotes</th>
      <th>person_id</th>
      <th>primary_name</th>
      <th>primary_profession</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>tt0069049</td>
      <td>The Other Side of the Wind</td>
      <td>The Other Side of the Wind</td>
      <td>2018</td>
      <td>122.0</td>
      <td>Drama</td>
      <td>6.9</td>
      <td>4517</td>
      <td>nm0000080</td>
      <td>Orson Welles</td>
      <td>actor,director,writer</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Action</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Animation</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>30</th>
      <td>tt0249516</td>
      <td>Foodfight!</td>
      <td>Foodfight!</td>
      <td>2012</td>
      <td>91.0</td>
      <td>Comedy</td>
      <td>1.9</td>
      <td>8248</td>
      <td>nm0440415</td>
      <td>Lawrence Kasanoff</td>
      <td>producer,writer,director</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>55</th>
      <td>tt0293069</td>
      <td>Dark Blood</td>
      <td>Dark Blood</td>
      <td>2012</td>
      <td>86.0</td>
      <td>Thriller</td>
      <td>6.6</td>
      <td>1053</td>
      <td>nm0806293</td>
      <td>George Sluizer</td>
      <td>director,producer,writer</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
</div>




```python
imdbnew_general_directors = imdbnew[(imdbnew['rating'] == 'High') & (imdbnew['numvotes'] > 5000)]
```


```python
imdb_general_directors = imdbnew_general_directors.primary_name.value_counts()[:10].to_frame()
imdb_general_directors
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
      <th>primary_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Christopher Nolan</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Can Ulkay</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Dragan Bjelogrlic</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Joe Russo</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Denis Villeneuve</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Lee Unkrich</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Anthony Russo</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Martin Scorsese</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Sujoy Ghosh</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Joshua Oppenheimer</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Filtering rows where rating is 'High' and number of votes greater than 5000 for drama category
imdbnew_drama_directors = imdbnew[(imdbnew['rating'] == 'High') & (imdbnew['numvotes'] > 5000) & (imdbnew['genres'] == 'Drama')]
```


```python
#Getting the top 5 directors for the drama category
imdb_drama_directors = imdbnew_drama_directors.primary_name.value_counts()[:5].to_frame()
imdb_drama_directors
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
      <th>primary_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>S.S. Rajamouli</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Denis Villeneuve</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Zoya Akhtar</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Quentin Tarantino</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Nuri Bilge Ceylan</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Filtering rows where rating is 'High' and number of votes greater than 5000 for comedy category
imdbnew_comedy_directors = imdbnew[(imdbnew['rating'] == 'High') & (imdbnew['numvotes'] > 5000) & (imdbnew['genres'] == 'Comedy')]
```


```python
#getting the top 5 directors for the comedy category
imdb_comedy_directors = imdbnew_comedy_directors.primary_name.value_counts()[:5].to_frame()
imdb_comedy_directors
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
      <th>primary_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dragan Bjelogrlic</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Lee Unkrich</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Alphonse Puthren</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Adrian Molina</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Damián Szifron</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Filtering rows where rating is 'High' and number of votes greater than 5000 for action category
imdbnew_action_directors = imdbnew[(imdbnew['rating'] == 'High') & (imdbnew['numvotes'] > 5000) & (imdbnew['genres'] == 'Action')]
```


```python
#getting the top 5 directors for the comedy category
imdb_action_directors = imdbnew_action_directors.primary_name.value_counts()[:5].to_frame()
imdb_action_directors
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
      <th>primary_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Christopher Nolan</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Joe Russo</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Sukumar</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Anthony Russo</th>
      <td>2</td>
    </tr>
    <tr>
      <th>S.S. Rajamouli</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 8.4. Relationship between domestic gross and worldwide gross using bom_movies

Here I will use the correlation coefficient measures the strength and relationship between two variables


```python
bom_movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3356 entries, 0 to 3386
    Data columns (total 5 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           3356 non-null   object 
     1   studio          3356 non-null   object 
     2   domestic_gross  3356 non-null   float64
     3   foreign_gross   3356 non-null   float64
     4   year            3356 non-null   int64  
    dtypes: float64(2), int64(1), object(2)
    memory usage: 157.3+ KB
    


```python
correlation = bom_movies['domestic_gross'].corr(bom_movies['foreign_gross'])
correlation
```




    0.792348350882411




```python
#Visualizing using a scatter plot
plt.scatter(bom_movies['domestic_gross'], bom_movies['foreign_gross'])
plt.xlabel('Domestic Gross')
plt.ylabel('Worldwide Gross')
plt.show()
```


    
![png](Movie_Analysis_files/Movie_Analysis_177_0.png)
    


Observations:
- There appears to be a high positive correlation (0.792348350882411) between domestic and foreign gross meaning that as the domestic gross increases, the worldwide gross also increases.
- Majority of the dots are clustered in the bottom left corner of the graph, indicating that most movies have relatively low domestic and worldwide grosses.

## 8.5 Most successful studios

I will gather this data from bom_movies dataframe

- What is the relationship between domestic gross and worldwide gross.
- Sum up the domestic_gross and foreign_gross columns to form a new column called total_gross
- Group by studio and total_gross


```python
bom_movies.head(5)
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
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415000000.0</td>
      <td>652000000.0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334200000.0</td>
      <td>691300000.0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296000000.0</td>
      <td>664300000.0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292600000.0</td>
      <td>535700000.0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238700000.0</td>
      <td>513900000.0</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>




```python
bom_movies['total_gross'] = bom_movies.domestic_gross + bom_movies.foreign_gross
```


```python
bom_studios = bom_movies.groupby(['studio'])['total_gross'].sum().to_frame()

```


```python
bom_studios['total_gross'] = bom_studios['total_gross'] / 1e7
```


```python
bom_studios = bom_studios.sort_values('total_gross',ascending = False)[:5]
bom_studios 


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
      <th>total_gross</th>
    </tr>
    <tr>
      <th>studio</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BV</th>
      <td>4421.288390</td>
    </tr>
    <tr>
      <th>Fox</th>
      <td>3100.536660</td>
    </tr>
    <tr>
      <th>WB</th>
      <td>3083.594900</td>
    </tr>
    <tr>
      <th>Uni.</th>
      <td>2975.716419</td>
    </tr>
    <tr>
      <th>Sony</th>
      <td>2240.491910</td>
    </tr>
  </tbody>
</table>
</div>




```python
#plotting a bar graph of Most Successful Studios
fig,ax = plt.subplots(figsize=(15,10))

title = 'Most Successful Studios'
y_label = 'Total Gross Earnings (Million USD)'
x_label = 'Studio'

bom_studios.total_gross.plot(kind='bar')
ax.set_title(title,fontsize=15)
ax.set_ylabel(y_label,fontsize=15)
ax.set_xlabel(x_label,fontsize=15)
plt.xticks(rotation = 60,fontsize=15)
fig.savefig('Images/Most Successful Studios.png');
```


    
![png](Movie_Analysis_files/Movie_Analysis_186_0.png)
    



```python

```
