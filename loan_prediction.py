{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "separated-fusion",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing all the necessary modules\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt\n",
    "import seaborn as sns\n",
    "get_ipython().run_line_magic('matplotlib', 'inline')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "reserved-convention",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import preprocessing\n",
    "from sklearn import metrics\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "mobile-prerequisite",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "reported-administrator",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score,f1_score\n",
    "from sklearn.model_selection import cross_val_score\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "dietary-suffering",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Loading the dataframe\n",
    "df=pd.read_csv(\"D:\\Misc\\Projects_ML\\loan_data_set.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "greater-poker",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',\n",
       "       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',\n",
       "       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the columns\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "variable-guess",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001002</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001003</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>N</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001005</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001006</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001008</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married Dependents     Education Self_Employed  \\\n",
       "0  LP001002   Male      No          0      Graduate            No   \n",
       "1  LP001003   Male     Yes          1      Graduate            No   \n",
       "2  LP001005   Male     Yes          0      Graduate           Yes   \n",
       "3  LP001006   Male     Yes          0  Not Graduate            No   \n",
       "4  LP001008   Male      No          0      Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             5849                0.0         NaN             360.0   \n",
       "1             4583             1508.0       128.0             360.0   \n",
       "2             3000                0.0        66.0             360.0   \n",
       "3             2583             2358.0       120.0             360.0   \n",
       "4             6000                0.0       141.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area Loan_Status  \n",
       "0             1.0         Urban           Y  \n",
       "1             1.0         Rural           N  \n",
       "2             1.0         Urban           Y  \n",
       "3             1.0         Urban           Y  \n",
       "4             1.0         Urban           Y  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "confirmed-edmonton",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dropping the unnecessary columns before the train test split\n",
    "df.drop(['Loan_ID', 'Gender', 'Dependents', 'Married'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "agricultural-carol",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education             0\n",
       "Self_Employed        32\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           22\n",
       "Loan_Amount_Term     14\n",
       "Credit_History       50\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the null values after droping unnecessary columns\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "horizontal-voltage",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Replace the null values of self employed\n",
    "df.Self_Employed.fillna('No', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "external-demonstration",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Replace the null values for Loan amount and loan amount term with their respective mean\n",
    "df.LoanAmount.fillna(df.LoanAmount.mean(), inplace=True)\n",
    "df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "bacterial-beach",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Replace the null values for credit history\n",
    "df['Credit_History'].value_counts()\n",
    "df.Credit_History.fillna(0.0, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "chicken-preparation",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Final check,if there are any null values\n",
    "df.isnull().sum().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "enabling-transcript",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education            Graduate\n",
       "Self_Employed              No\n",
       "ApplicantIncome         81000\n",
       "CoapplicantIncome           0\n",
       "LoanAmount                360\n",
       "Loan_Amount_Term          360\n",
       "Credit_History              0\n",
       "Property_Area           Rural\n",
       "Loan_Status                 N\n",
       "Name: 409, dtype: object"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#So there are no null values in the present features\n",
    "#Checking the maximum and minimum amount of the Applicant income\n",
    "df.loc[df['ApplicantIncome'].idxmax()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "oriented-corpus",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education            Graduate\n",
       "Self_Employed              No\n",
       "ApplicantIncome           150\n",
       "CoapplicantIncome        1800\n",
       "LoanAmount                135\n",
       "Loan_Amount_Term          360\n",
       "Credit_History              1\n",
       "Property_Area           Rural\n",
       "Loan_Status                 N\n",
       "Name: 216, dtype: object"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[df['ApplicantIncome'].idxmin()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "veterinary-samuel",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Using Label encoding for multiple columns\n",
    "colums_in_list=df.columns.tolist()\n",
    "le = LabelEncoder()\n",
    "for col in colums_in_list:\n",
    "    if df[col].dtype==object:\n",
    "         df[col] = le.fit_transform(df[col])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "instant-webcam",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>146.412162</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.000000</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.000000</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.000000</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Education  Self_Employed  ApplicantIncome  CoapplicantIncome  LoanAmount  \\\n",
       "0          0              0             5849                0.0  146.412162   \n",
       "1          0              0             4583             1508.0  128.000000   \n",
       "2          0              1             3000                0.0   66.000000   \n",
       "3          1              0             2583             2358.0  120.000000   \n",
       "4          0              0             6000                0.0  141.000000   \n",
       "\n",
       "   Loan_Amount_Term  Credit_History  Property_Area  Loan_Status  \n",
       "0             360.0             1.0              2            1  \n",
       "1             360.0             1.0              0            0  \n",
       "2             360.0             1.0              2            1  \n",
       "3             360.0             1.0              2            1  \n",
       "4             360.0             1.0              2            1  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "arranged-generic",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education              int32\n",
       "Self_Employed          int32\n",
       "ApplicantIncome        int64\n",
       "CoapplicantIncome    float64\n",
       "LoanAmount           float64\n",
       "Loan_Amount_Term     float64\n",
       "Credit_History       float64\n",
       "Property_Area          int32\n",
       "Loan_Status            int32\n",
       "dtype: object"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "korean-singing",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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fPn5J4TmykVISAgt32hLwUKF/FLGs6F6zdosnD+He+9oRFhYGG+1P1lPzR64i0HDPSN3Xn2jLe91eNv7kwq1fNpUh3RtqgMAJUtdStXqtWh2/52YhdD4jrv/88OW/wvMdw5EsMnkpwKmOefeMLObgI+AQ8Bc4FLn3K1mlg/4BjgfWAxUBxo55zaa2aPA63iiYsudc83M7HagJ7AbTwevinPuejO7HBgDpAAvAvWBA8657t5hm32B84A/gcecc7vN7Ec8P1mwzMwigWXOuZKnK1++q14I3v+8s2j19G7+zkLAiCn4b4PH/w37jxz3dxYCQpH8wfMglOy295DaVFYEWX86WyWn6CtCVuQK0SC0rIoqkCsgzsB8dTvmWOM/PKttjtdJUEfenHOZzjR3zk0DMvwIjXPuMNDwFNsMxjNHzXfZN3g6e+nTbgCu9Fk012fdL0CGx5c55673eZ3IKea8iYiIiIjIf1NQd95EREREROQ/RA8sEREREREREX9T5E1ERERERIJDkE+OVeRNREREREQkACjyJiIiIiIiwUFz3kRERERERMTfFHkTEREREZHgoDlvIiIiIiIi4m+KvImIiIiISHDQnDcRERERERHxN0XeREREREQkOGjOm4iIiIiIiPibIm8iIiIiIhIcNOdNRERERERE/E2dNxERERERkQCgYZMiIiIiIhIc9MASERERERER8TdF3kREREREJDjogSUiIiIiIiLib4q8iYiIiIhIcNCcNxEREREREfE3Rd5ERERERCQ4aM6biIiIiIiI+JsibyIiIiIiEhwUeRMRERERERF/U+QtgK2e3s3fWQgIFRq+7u8sBIyZozv5OwsBIaZQmL+zEBA2J+71dxYCRrHCalNZkZSc4u8sBIxcobo/nxWHko/7OwsBI6pAgHQb9LRJERERERER8bcA6UKLiIiIiIicgea8iYiIiIiIiL8p8iYiIiIiIsFBc95ERERERETE39R5ExERERERCQAaNikiIiIiIsFBDywRERERERERf1PkTUREREREgoMeWCIiIiIiIiL+psibiIiIiIgEBVPkTURERERERPxNkTcREREREQkKiryJiIiIiIiI3ynyJiIiIiIiwSG4A2+KvImIiIiIiAQCRd5ERERERCQoaM6biIiIiIiI+J0ibyIiIiIiEhQUeRMRERERERG/U+RNRERERESC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vGXrj3SLiIiIiIjIWaLIm4iIiIiIBIVgn/OmyJuIiIiIiEgAUORNRERERESCgiJvIiIiIiIi4neKvImIiIiISFAI8sCbIm8iIiIiIiKBQJE3EREREREJCprzJiIiIiIiIn6nyJuIiIiIiASFIA+8KfImIiIiIiISCNR5ExERERERCQAaNikiIiIiIkFBDywRERERERERv1PkTUREREREgkKQB94UeRMREREREQkEiryJiIiIiEhQ0Jw3ERERERER8TtF3kREREREJCgEeeBNkTcREREREZFAoMibiIiIiIgEBc15ExEREREREb9T5C2A7T9y3N9ZCAgxhcL8nYWAsX1+L39nISCUqNnC31kICGOHtvN3FgJGSJDfKT5bwvPqa0tWHTia5O8sBAS1qeAT7JdTRd5EREREREQCgDpvIiIiIiISFMwsx/6ykJebzOxXM/vdzN44RZqmZrbWzNaY2fAz7VOxYhERERERkbPIzEKBT4AbgK3AUjOb6Jxb65OmNPAmUNM5t9vMos+0X3XeREREREQkKJxDc96uA353zv0JYGYjgduBtT5pngI+cc7tBnDOxZ9ppxo2KSIiIiIicnadD2zxeb/Vu8zX5cDlZjbfzBaZ2U1n2qkibyIiIiIiIv+QmTUHmvss6uec6/cPdpELKA1cD1wAzDGzis65PafbQEREREREJODl5I90eztqp+qsbQMu9Hl/gXeZr63AYufcceAvM9uApzO39FTH1LBJERERERGRs2spUNrMSplZHuA+YGK6NBPwRN0ws0g8wyj/PN1OFXkTEREREZGgcK48sMQ5l2RmLwDfAaHAQOfcGjN7F1jmnJvoXdfQzNYCycDrzrmdp9uvOm8iIiIiIiJnmXNuKjA13bJ2Pq8d0NL7lyXqvImIiIiISFDIyTlv/qA5byIiIiIiIgFAkTcREREREQkKiryJiIiIiIiI3ynyJiIiIiIiQSHIA2+KvImIiIiIiAQCRd5ERERERCQoaM6biIiIiIiI+J0ibyIiIiIiEhSCPPCmyJuIiIiIiEggUORNRERERESCgua8iYiIiIiIiN+p8yYiIiIiIhIANGxSRERERESCQpCPmlTkTUREREREJBAo8ian5Zyj94edWbRgLmFhYbzRrhOXl70iQ7pf162h87ttOHr0KNVqxPLiq2+kThgd9/Uwxo8ZSWhIKNVq1uaZ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ExERERERCQAaNikiIiIiIkEhuAdNKvImIiIiIiISEBR5C2BF8ufxdxYCwubEvf7OQsB4csRyf2chIIwd2s7fWQgIdz/0rr+zEDCWTe7i7ywEhPAwfW3JqvPyhPo7CwEhKcX5OwtylulHukVERERERMTvdAtLRERERESCQkhwB94UeRMREREREQkEiryJiIiIiEhQ0Jw3ERERERER8TtF3kREREREJCgEeeBNkTcREREREZFAoMibiIiIiIgEBc15ExEREREREb9T5E1ERERERIKCfudNRERERERE/E6RNxERERERCQqa8yYiIiIiIiJ+p86biIiIiIhIANCwSRERERERCQrBPWhSkTcREREREZGAoMibiIiIiIgEhRA9sERERERERET8TZE3EREREREJCkEeeFPkTUREREREJBAo8iYiIiIiIkFBP9ItIiIiIiIifqfIm4iIiIiIBIUgD7wp8iYiIiIiIhIIFHkTEREREZGgEOy/86bOm2SZc44uH7zHvDmzCcsXRsf3OlPuivIZ0vXu1ZNJEyewb+8+Fi1bnrq8W+f3WbpkMQCHjxxh966dzFu0LMfyn1NWLlvI0M8/JCUlhTo33k7jpo+mWb9+1c8M69eTLX/9znNvdOK6WvUB2PTHBgZ90pkjhw4SEhJK43sfo1qdG/xRhBxT5eLCvFC7JCFmTF0Tx4iftqdZf2O5KJ6udTGJB44BMGHlDqauiafyBQV5LrZkarqLiuSj47QNzP9zd05mP8es+3kx4wf2wqWkULXBrTS466E06/9Y8wvjB37M35v+5OGW7alco27quklffcbanxYC0LDJo1zlbW//RX3bP0ij2hVI2LWfa5u87+/s5Lifl8xnYJ/upKQk0+DmO7nrgcfSrD9+7Bi9Orflzw3rKFCwMK+260x0sRL8tm41n/XoBHg+B+599GmqxdYDYPLY4Xw/ZTw4R4Nb7qTxPQ/meLnONuccn/bswtKFc8kbFsZrbTpSuswVGdJtWL+W7p3acOzoUapUj+W5V1pjZgzq14eFc2dhISEULlyU19t0JCIqmhU/L6V96xYUK3E+ALXq1Oehx5/J6eKdVc45PunRhcUL55I3bxit2nbk8rKZ11XXjm04evQoVavH8nzL1mkeKjFq2GA+7/0h46bNplDhIvzy01LatfKpq+vr88gTgVtXJ9rUkgWeNvV621O3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IiIiIlI8Zd5ERERERCQpJHniTZk3ERERERGRRKDMm4iIiIiIJIWUJB/0psybiIiIiIhIAlDmTUREREREkkJy592UeRMREREREUkIaryJiIiIiIgkAHWbFBERERGR5JDk/SaVeRMREREREUkAyryJiIiIiEhSsCRPvSnzJiIiIiIikgCUeRMRERERkaSQ5Gt0K/MmIiIiIiKSCJR5ExERERGRpJDkiTdl3kRERERERBKBMm8iIiIiIpIckjz1psybiIiIiIhIAlDmTUREREREkoLWeRMREREREZHIKfOWwHI2etQhJIQyqbpHUWIbN0YdQUJYsXZD1CEkhAo7pEYdQsKoUl4fxyUx45OHow4hYdQ7/raoQ0gI37z/YNQhyFamdd5EREREREQkcrrVJyIiIiIiSSHJE2/KvImIiIiIiCQCNd5EREREREQSgLpNioiIiIhIckjyfpPKvImIiIiIiCQAZd5ERERERCQpaJFuERERERERiZwybyIiIiIikhS0SLeIiIiIiIhETpk3ERERERFJCkmeeFPmTUREREREJBEo8yYiIiIiIskhyVNvyryJiIiIiIgkAGXeREREREQkKWidNxEREREREYmcMm8iIiIiIpIUtM6biIiIiIiIRE6ZNxERERERSQpJnnhT5k1ERERERCQRKPMmIiIiIiLJIclTb8q8iYiIiIiIJAA13kRERERERBKAuk2KiIiIiEhSSPZFutV4k81yd57s3ZOJ48eQlpbGnV3vY5/96hUq98PM7+l5b2fWrl1Lw8bNuLH9nVi40MZbb7zK22++TmpKKg2bNOeam27ljwXzueT/zmC33esAUK/BgbTv2LU0D22rcnf6PvogkyeMoVxaGrd17s7e+xaupx9nzeDh+zqzbu1ajmjUjOtuuQMzo/+zfZgwZhSWkkK1ajW4vXN3amZkMuf33+h9fxd+/nEmba++kdYXti39g/uP3J1nHn+IyRPGUi4tjVvvupe99t2/ULmfZs3gkQe6hnXTlKvbdcDMWP73X/To2oFFfy4gc8ed6XhvLypXqcLyv//msR5388eCeeywww7c3PEe6uyxFwBtzz2Z8hUqkpqSQkpqGZ54YVBpH/Z/9vWkcTz3RC82btzI8aeeSes2l+V7fP26dTxyfxd++XEmlatUpUO3B6m10878/dcyena9nZ9mfU/Lk07nmlvuzHvO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DcBHQCrwort/b2b3Al+5+zDgBeAVM/sZWELQwNssNd5ERERERCQ5xNEi3e7+AfBBgX1dY35fA7T+J6+pbpMiIiIiIiIJQJk3ERERERFJCiVYfy2hKfMmIiIiIiKSAJR5ExERERGRpFCa67xFQZk3ERERERGRBKDMm4iIiIiIJIUkT7wp8yYiIiIiIpIIlHkTEREREZHkkOSpN2XeREREREREEoAabyIiIiIiIglA3SZFRERERCQpaJFuERERERERiZwybwmsTIra3iWxKmd91CFIkqlUTpfOktiw0aMOIWGsWpcTdQgJoefnv0QdQsL45v0How4hIRx66h1Rh5AwVn/bJ+oQSkSLdIuIiIiIiEjkdPtYRERERESSQpIn3pR5ExERERERSQTKvImIiIiISFLQmDcRERERERGJnDJvIiIiIiKSJJI79abMm4iIiIiISAJQ5k1ERERERJKCxryJiIiIiIhI5JR5ExERERGRpJDkiTdl3kRERERERBKBMm8iIiIiIpIUNOZNREREREREIqfGm4iIiIiISAJQt0kREREREUkKluRTlijzJiIiIiIikgCUeRMRERERkeSQ3Ik3Zd5EREREREQSgTJvIiIiIiKSFJI88abMm4iIiIiISCJQ5k1ERERERJKCFukWERERERGRyCnzJiIiIiIiSUHrvImIiIiIiEjklHkTEREREZHkkNyJNzXeZPPcnccf7sGEcV+Qllaeu7rdz7771StUbtbM73mgWyfWrl1DoybNaXdbR8yMn36YSa8e97Ju3VpSU8vQ/o7O1GtwIADffPUlTzzSkw0bNlCtWnX6PDugtA/vP3F3nnn8ISZPHEu5cmncete97LXv/oXK/fTDDB55oCvr1q7liIZNubpdB8yM5X//RY+7O7DozwVk7rgzHe/tReXKVZgwZhSvPN+XlBQjJbUMV990O/UPPIQp30zmuSd75b3u3Dm/c8fdPWnc/NjSPOx/ZVvVVa4fZ07n1msv4c67e9L0mOMTuq5yuTtPPtKTSePHkJaWxh1d7mOfIs69H2Z+z4PdO7N27VqOatyMG2+9EzOj/3N9ef/doVStVh2AK669iYZNmrN+/Xoe6XEPP8z6HrMUbrz1Tg4+7IjSPrytxt3p++iDfDl+DOXS0ri9S3f23rdwPf04awa9undm3dq1HNm4GdfdckdQT8/0YfyYUVhK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\n",
      "text/plain": [
       "<Figure size 1080x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Correlation of the features\n",
    "corr = df.corr()\n",
    "plt.figure(figsize=(15,10))\n",
    "sns.heatmap(corr, annot=True, cmap='Blues')\n",
    "plt.title(\"Feature Correlation Heatmap\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "scheduled-effects",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Preparing the input and output \n",
    "X = df.drop(\"Loan_Status\", axis=1)\n",
    "Y= df[\"Loan_Status\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "meaningful-ecuador",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting the train and test set\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=30) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "earlier-fiber",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Logistic Regression\n",
    "model_lr= LogisticRegression()\n",
    "model_lr.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "regional-luther",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Decision tree\n",
    "model_decisiontree=DecisionTreeClassifier()\n",
    "model_decisiontree.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "maritime-palestinian",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Random Forest\n",
    "model_randomforest= RandomForestClassifier(n_estimators=100)\n",
    "model_randomforest.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "induced-catch",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVC()"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Suppart Vector Machine (SVC)\n",
    "model_SVC=SVC()\n",
    "\n",
    "model_SVC.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "perceived-cutting",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\anaconda3\\envs\\workenv\\lib\\site-packages\\xgboost\\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[21:33:02] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.300000012, max_delta_step=0, max_depth=6,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#XGB classifier\n",
    "model_XGB=XGBClassifier()\n",
    "model_XGB.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "celtic-celebrity",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression accuracy is  74.5945945945946\n"
     ]
    }
   ],
   "source": [
    "#Checking the accuracy in Logistic Regression\n",
    "\n",
    "acc_lr=model_lr.score(X_test,Y_test)\n",
    "print(\"Logistic Regression accuracy is \",format(acc_lr*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "hydraulic-indonesia",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree accuracy is  63.24324324324324\n"
     ]
    }
   ],
   "source": [
    "#Checking the accuracy in Decision Tree\n",
    "acc_dt=model_decisiontree.score(X_test,Y_test)\n",
    "print(\"Decision Tree accuracy is \",format(acc_dt*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "vocational-profit",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random forest accuracy is  70.27027027027027\n"
     ]
    }
   ],
   "source": [
    "#Random Forest\n",
    "model_randomforest= RandomForestClassifier(n_estimators=100)\n",
    "model_randomforest.fit(X_train, Y_train)\n",
    "acc_rf=model_randomforest.score(X_test,Y_test)\n",
    "print(\"Random forest accuracy is \",format(acc_rf*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "alternate-update",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC accuracy is  71.89189189189189\n"
     ]
    }
   ],
   "source": [
    "#Checking the accuracy in Suppart Vector Machine\n",
    "acc_svc=model_SVC.score(X_test,Y_test)\n",
    "print(\"SVC accuracy is \",format(acc_svc*100))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "rural-microwave",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBClassifier accuracy is  73.51351351351352\n"
     ]
    }
   ],
   "source": [
    "#Checking the accuracy in Random Forest\n",
    "acc_XGB=model_XGB.score(X_test,Y_test)\n",
    "print(\"XGBClassifier accuracy is \",format(acc_XGB*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "imported-penny",
   "metadata": {},
   "outputs": [],
   "source": [
    "# It can be seen that the Logistic regression has high accuracy compared to other classifiers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "swedish-pipeline",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
