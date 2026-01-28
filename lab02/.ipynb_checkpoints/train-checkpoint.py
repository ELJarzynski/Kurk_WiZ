import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Settings of terminal setup"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('train.csv')

print(df.head())
print(df.tail())
print(df.shape)
print(df.shape[0])
print(df.shape[1])

print(df.dtypes)
print(df.count)
print(df.describe())
print(df.describe(include='all'))
print(df.info())
print(df.duplicated())

df.drop_duplicates()
df.drop_duplicates(subset='User_ID', inplace=False)


p0=df.Purchase.min()
p100=df.Purchase.max()
q1=df.Purchase.quantile(0.25)
q2=df.Purchase.quantile(0.5)
q3=df.Purchase.quantile(0.75)
iqr=q3-q1

lc = q1 - 1.5*iqr
uc = q3 + 1.5*iqr


print(lc, uc)

print( "p0 = " , p0 ,", p100 = " , p100 ,", lc = " , lc ,", uc = " , uc)
df.Purchase.plot(kind='box')
df.Purchase.clip(upper=uc)
df.Purchase.clip(upper=uc,inplace=True)
df.Purchase.plot(kind='box')
print( "p0 = " , p0 ,", p100 = " , p100 ,", lc = " , lc ,", uc = " , uc)
df.Purchase.plot(kind='box')

df.Purchase.clip(upper=uc)
df.Purchase.clip(upper=uc,inplace=True)
df.Purchase.plot(kind='box')


df.isna()

df.isna().sum()/df.shape[0]
print(df.Product_Category_2.mode()[0])
df.Product_Category_2.fillna(df.Product_Category_2.mode()[0],inplace=True)
df.isna().sum()
df.dropna(axis=1,inplace=True)
print(df.dtypes)

df.Purchase.hist()
plt.show()
df.Purchase.plot(kind='hist' , grid = True)
plt.show()

plt.hist(df.Purchase)
plt.grid(True)
plt.show()

df.Purchase.plot(kind='box')
plt.show()

plt.boxplot(df.Purchase)
plt.show()

df.groupby('Gender').City_Category.count().plot(kind='pie')
plt.show()
sns.countplot(df.Marital_Status)
plt.show()
df.groupby('City_Category').City_Category.count().plot(kind='pie')
plt.show()
sns.countplot(df.Age)
plt.show()
df.groupby('Stay_In_Current_City_Years').City_Category.count().plot(kind='pie')
plt.show()
sns.countplot(df.Occupation)
plt.show()
df.groupby('Product_Category_1').City_Category.count().plot(kind='barh')
plt.show()
df.plot(x='Product_Category_1',y='Product_Category_2',kind = 'scatter')
plt.show()
df.select_dtypes(['float64' , 'int64']).corr()

sns.heatmap(df.select_dtypes(['float64' , 'int64']).corr(),annot=True)
plt.show()

df.groupby('Occupation').Purchase.sum().plot(kind='bar')
plt.show()
summary=df.groupby('Occupation').Purchase.sum()
plt.bar(x=summary.index , height=summary.values)
plt.show()

sns.barplot(x=summary.index , y=summary.values)
plt.show()
df.groupby('Age').Purchase.sum().plot(kind='line')
plt.show()

df.groupby('Gender').Purchase.sum().plot(kind='pie')
plt.show()

df.groupby('City_Category').Purchase.sum().plot(kind='area')
plt.show()
df.groupby('Stay_In_Current_City_Years').Purchase.sum().plot(kind='barh')
plt.show()

sns.boxplot(x='Marital_Status',y='Purchase',data=df)
plt.show()
print(pd.crosstab(df.Age,df.Gender))

sns.heatmap(pd.crosstab(df.Age,df.Gender))
plt.show()
sns.heatmap(pd.crosstab(df.City_Category,df.Stay_In_Current_City_Years))
plt.show()
