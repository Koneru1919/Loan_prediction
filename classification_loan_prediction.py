#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("D:\Misc\Projects_ML\loan_data_set.csv")


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


#Dropping the unnecessary columns before the train test split
df.drop(['Loan_ID', 'Gender', 'Dependents', 'Married'], axis=1, inplace=True)


# In[6]:


df.head()


# In[7]:


#Checking the null values after droping unnecessary columns
df.isnull().sum()


# In[8]:


#Replace the null values of self employed
df.Self_Employed.fillna('No', inplace=True)


# In[9]:


#Replace the null values for Loan amount and loan amount term with their respective mean
df.LoanAmount.fillna(df.LoanAmount.mean(), inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(), inplace=True)


# In[10]:


#Replace the null values for credit history
df['Credit_History'].value_counts()
df.Credit_History.fillna(0.0, inplace=True)


# In[11]:


#Final check,if there are any null values
df.isnull().sum().any()


# In[12]:


#So there are no null values in the present features 


# In[13]:


#Checking the maximum and minimum amount of the Applicant income
df.loc[df['ApplicantIncome'].idxmax()]


# In[14]:


df.loc[df['ApplicantIncome'].idxmin()]


# In[15]:


df.columns


# In[16]:


#Label encoding multiple columns
colums_in_list=df.columns.tolist()
le = LabelEncoder()
for col in colums_in_list:
    if df[col].dtype==object:
         df[col] = le.fit_transform(df[col])


# In[17]:


df.head()


# In[18]:


df.dtypes


# In[19]:


#Correlation of the features
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap='Blues')

plt.title("Feature Correlation Heatmap")
plt.show()


# In[20]:


#Preparing the input and output 
X = df.drop("Loan_Status", axis=1)
Y= df["Loan_Status"]


# In[21]:


#Splitting the train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30) 


# In[22]:


#Logistic Regression
model_lr= LogisticRegression()
model_lr.fit(X_train,Y_train)


# In[23]:


#Decision tree
model_decisiontree=DecisionTreeClassifier()
model_decisiontree.fit(X_train,Y_train)


# In[24]:


#Random Forest
model_randomforest= RandomForestClassifier(n_estimators=100)

model_randomforest.fit(X_train, Y_train)


# In[25]:



#Suppart Vector Machine (SVC)
model_SVC=SVC()

model_SVC.fit(X_train,Y_train)


# In[26]:


#XGB classifier
model_XGB=XGBClassifier()
model_XGB.fit(X_train, Y_train)


# In[27]:


#Checking the accuracy in Logistic Regression

acc_lr=model_lr.score(X_test,Y_test)

print("Logistic Regression accuracy is ",format(acc_lr*100))


# In[28]:


#Checking the accuracy in Decision Tree
acc_dt=model_decisiontree.score(X_test,Y_test)
print("Decision Tree accuracy is ",format(acc_dt*100))


# In[29]:


#Random Forest
model_randomforest= RandomForestClassifier(n_estimators=100)

model_randomforest.fit(X_train, Y_train)

acc_rf=model_randomforest.score(X_test,Y_test)
print("Random forest accuracy is ",format(acc_rf*100))


# In[30]:


#Checking the accuracy in Suppart Vector Machine
acc_svc=model_SVC.score(X_test,Y_test)
print("SVC accuracy is ",format(acc_svc*100))


# In[31]:


#Checking the accuracy in Random Forest
acc_XGB=model_XGB.score(X_test,Y_test)
print("XGBClassifier accuracy is ",format(acc_XGB*100))


# In[32]:


# It can be seen that the Logistic regression has high accuracy compared to other classifiers

