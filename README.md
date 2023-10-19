# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Name: PRAVINRAJJ G.K
Reg.no: 212222240080
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
## Dataset
![263522299-0833a37c-71ed-4f5e-9ecd-c57a6c8fcacf](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/57da4e99-c537-49a6-9d1c-6b9ec83ce0fe)

## Dropping unwanted features
![263522542-ce2f4978-5bc6-408f-ae70-f00114bfead5](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/c23fb046-4770-436a-93eb-62240797196a)

## Checking for null values
![263522689-92e62941-5191-444d-84bd-237e66d038ab](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/bc19a073-d993-4358-a9e4-2a7807c20eb7)

## Checking for duplication
![263522781-07dc5041-4f55-4d9a-ba6d-1d7588f79374](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/b84808a6-1a99-45c9-9243-e2403e24c9a4)

## Describing the dataset
![263522956-8f74cfd8-9476-48d1-9e48-77c008c7591b](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/412595a1-a219-4c5c-b4e9-6222ed4a75a6)

## Scaling the values
![263523018-10d1fbb4-47bc-4da4-adc9-fd6591dcd9b2](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/2c378e3c-4398-4505-86a5-4facad73ec35)

## X features
![263523137-31b79862-9f30-4deb-aaf5-4b8b9508b822](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/7f4f3f1a-efbf-442f-b995-3007e0f4202e)

## Y fetures
![263523159-139305ad-0c35-4788-acc7-f72a90757d42](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/e7e8e396-d09f-4e02-a4c2-36500b6290ad)

## Splitting the training testing dataset
![263523228-8daa270e-6b98-470c-b2e0-2529d60c000b](https://github.com/Pravinrajj/Ex.No.1---Data-Preprocessing/assets/117917674/0ef313a0-ade7-4f1b-83ed-3b489ecd38ee)


## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle
