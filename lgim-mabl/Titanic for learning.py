
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt = pd.read_csv('E:/python/pratice/kaggel pratice/titanic.csv')
print(dt.columns.values)
dt.head()
dt.tail()
dt.info()
 # is used to find the max,mean,std deviation,min and to know how many rows are not having data in the columns 
dt.describe()
dt.shape
print(dt.dtypes)
dt.isnull()  # is used to find the null values in the x

dt["Survived"].value_counts() # to know the  categories in columns

dt['Embarked'].value_counts()
sns.heatmap(dt.isnull(),yticklabels=False) #to show the graphical representation how many values are null in which column
sns.set_style('whitegrid')# to plot the grid with columns
sns.countplot(dt["Survived"])#
#sns.countplot(dt="Survivied",data=train)# imported sea bord to plot the  values to know th how many are died/live

sns.set_style("whitegrid")
sns.countplot(dt["Survived"],hue =dt['Sex']) #to plot the graph repect to sex bases

sns.set_style("whitegrid")
sns.countplot(dt["Survived"],hue=dt["Pclass"]) #

sns.distplot(dt["Age"].dropna(),kde=False,color="darkred",bins=40) # distrubution plot of ages kernal distru is flase


sns.countplot(dt["SibSp"])

# to fill the null values in the age with respect to pclass values 
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=dt,palette='winter')

dt["Age"]=dt[["Age","Pclass"]].apply(impute_age,axis=1)

def impute_age(cols):
    Age=cols[0]
     
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        if Pclass==2:
            return 29
        if Pclass==3:
            return 24
    else:
        return Age  
sns.set_style("whitegrid")
sns.heatmap(dt.isnull(),yticklabels=False)# to plot the null values

sex=pd.get_dummies(dt["Sex"],drop_first=True).head()
embark=pd.get_dummies(dt["Embarked"],drop_first=True).head()
dt.drop(["Sex","Embarked"],axis=1,inplace=True)
dt=pd.concat([dt,sex,embark],axis=1)
print(dt.columns.values)
print(dt.head())
X=dt.drop(["survived"],axis=1)



    
   
            


#filling the null values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X[:,2:3]) 
X[:,2:3]=imputer.transform(X[:,2:3])


X=X.fillna(X.mode().iloc[:,6])


imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer=imputer.fit(X.fillna(X.mode().iloc[:,6],inplace=True)) 
X[:,6]=imputer.transform(X.fillna(X.mode().iloc[:,6],inplace=True))

dt.info()


# labelencoder encoding the categorical values 

from sklearn.preprocessing import LabelEncoder
labelencoder_X= LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])


print(np.corrcoef(dt.Pclass,X.Age))

plt.scatter(X.Age,dt.Pclass)
print(X)
print(X.dtypes)
#X=fit_transform(X[])

from sklearn.preprocessing import LabelEncoder
labelencoder_X= LabelEncoder()
X.iloc[:,-1]= labelencoder_X.fit_transform(X.iloc[:,-1])



# correlation between columns and dependent variable
print(np.corrcoef(dt.Pclass,dt.Survived))
plt.scatter(dt.Pclass,dt.Survived)



