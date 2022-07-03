import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from datetime import date
import missingno as msno
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

!pip install catboost
!pip install lightgbm
!pip install xgboost
import warnings
from lightgbm import LGBMClassifier
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)
np.random.seed(1)

df=pd.read_excel("UC01_TV_raw_mini_dataset.xlsx")
df1=df.copy()


df.info()
df.index
df.shape
df.isnull().sum()

df.columns
df.nunique()
df["MATL_GROUP"].value_counts()
df["MATL_GROUP"].nunique()

sns.countplot(x=df["MATL_GROUP"], data=df)
plt.show()

##### Data_Preprocessing ####

#deletion of specified columns
df.drop(["ZCTYCODE","ZZ0040"],axis=1 ,inplace=True ) #deleted these 2 variable before because I wanted to have dataset which it doesnt have missing values.
df.shape
df.isnull().sum()

df["PRICE_GRP"] = df["PRICE_GRP"].astype("object")
df.dtypes

msno.bar(df)
plt.show()


####### mode assigned to "PRICE_GRP" and "CUST_GROUP" and "BOLGE"
df["PRICE_GRP"]=df["PRICE_GRP"].fillna(df["PRICE_GRP"].mode()[0])
df["PRICE_GRP"].isnull().sum()

df["CUST_GROUP"]=df["CUST_GROUP"].fillna(df["CUST_GROUP"].mode()[0])
df["CUST_GROUP"].isnull().sum()

df["BOLGE"]=df["BOLGE"].fillna(df["BOLGE"].mode()[0])
df["BOLGE"].isnull().sum()
#######

df.drop(["HZMBSLK"],axis=1 ,inplace=True )
df.drop(["BKM_ITEM"],axis=1 ,inplace=True )
df.drop(["ARZGRUP"],axis=1 ,inplace=True )


#this variable might be valuable if we consider all dataset.In this dataset all observations same in "ZPRDHYR3"
df.drop(["ZPRDHYR3"],axis=1 ,inplace=True )
df.drop(["ZSTCKODU"],axis=1 ,inplace=True )
df.drop(["ZURNGRP"],axis=1 ,inplace=True )

df.shape #currently (3590,21)

df.head(5)

# The format of columns "CRM_WYBEGD","CRMPOSTDAT","CRM_WYENDD" and "ZURTMONTH" have been translated into history.
df["CRM_WYBEGD"] = pd.to_datetime(df["CRM_WYBEGD"], format="%Y%m%d")
df["CRMPOSTDAT"] = pd.to_datetime(df["CRMPOSTDAT"], format="%Y%m%d")
df["CRM_WYENDD"] = pd.to_datetime(df["CRM_WYENDD"], format="%Y%m%d")
df["ZURTMONTH"] = pd.to_datetime(df["ZURTMONTH"], format="%Y%m")

# Feature Extraction as "WarrancyPERIOD"
df["WarrancyPERIOD"] = df["CRM_WYENDD"].dt.year - df["CRM_WYBEGD"].dt.year

# Feature Extraction as "URNLIFETIME".New variable explains items's lifetime value until breakdown.
df["URNLIFETIME"] = df["CRMPOSTDAT"].dt.year - df["ZURTMONTH"].dt.year

type("WarrancyPERIOD")
df["WarrancyPERIOD"]=df["WarrancyPERIOD"].astype("int32")
df["URNLIFETIME"]=df["URNLIFETIME"].astype("int32")

# We can extract one more variable with difference of "URNLIFETIME" and "WarrancyPERIOD"
# but I foresee their correlation will be high.That's why i didnt do.


# changing variables types to correct variable type
df.dtypes

df["PRICE_GRP"]=df["PRICE_GRP"].astype("object")
df["ZMLYGRP"]=df["ZMLYGRP"].astype("object")
df["ZRPRGRP"]=df["ZRPRGRP"].astype("object")
df["ZURNTIP"]=df["ZURNTIP"].astype("object")
df["MATL_GROUP"]=df["MATL_GROUP"].astype("object")
df.shape
#deleting variable which its format datetime

date_variables = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
df.drop(date_variables, axis=1,inplace=True)

df.head(5)
df.shape

df.dtypes


num_col=df[["WarrancyPERIOD","URNLIFETIME"]]
target=df["MATL_GROUP"]

#Analysis of the Target Variable with Numerical Variables
def target_summary_with_num(df, target, num_col):
    print(df.groupby(target).agg({num_col: "mean"}), end="\n\n\n")
    return


target_summary_with_num(df, "MATL_GROUP", "WarrancyPERIOD")
target_summary_with_num(df, "MATL_GROUP", "URNLIFETIME")
df.head(5)

import seaborn as sns

a = df.groupby("MATL_GROUP")["WarrancyPERIOD"].agg(lambda x: x.mean()).to_frame().reset_index()
ax = sns.barplot(x="MATL_GROUP", y="WarrancyPERIOD", data=a) #Average warranty period according to MATL_GROUP


b=df.groupby("MATL_GROUP")["URNLIFETIME"].agg(lambda x: x.mean()).to_frame().reset_index() #Average product life time according to MATL_GROUP
bx=sns.barplot(x="MATL_GROUP",y="URNLIFETIME",data=b)



df.head(10)
df["MATL_GROUP"].value_counts()

#Encoding
df.dtypes

for col in df.columns:
    df[col]=df[col].astype("str")

df['CUST_GROUP']=df['CUST_GROUP'].astype("str")
df['PRICE_GRP']=df['PRICE_GRP'].astype("str")
df['ZCRMPRD']=df['ZCRMPRD'].astype("str")
df['ZMLYGRP']=df['ZMLYGRP'].astype("str")
df['ZPRDHYR4']=df['ZPRDHYR4'].astype("str")
df['ZPRDHYR5']=df['ZPRDHYR5'].astype("str")
df['ZPRDHYR6']=df['ZPRDHYR6'].astype("str")
df['ZPRDHYR7']=df['ZPRDHYR7'].astype("str")
df['ZPRDHYR8']=df['ZPRDHYR8'].astype("str")
df['ZRPRGRP']=df['ZRPRGRP'].astype("str")
df['ZSIKAYET']=df['ZSIKAYET'].astype("str")
df['ZSIKAYET2']=df['ZSIKAYET2'].astype("str")
df['ZSIKAYET3']=df['ZSIKAYET3'].astype("str")
df['ZURNTIP']=df['ZURNTIP'].astype("str")
df['ZZMARKA']=df['ZZMARKA'].astype("str")
df['MATL_GROUP']=df['MATL_GROUP'].astype("str")
df["WarrancyPERIOD"]=df["WarrancyPERIOD"].astype("int64")
df["URNLIFETIME"]=df["URNLIFETIME"].astype("int64")

df.head(5)

label_encoder = preprocessing.LabelEncoder()

df['BOLGE']= label_encoder.fit_transform(df['BOLGE'])
df['CUST_GROUP']= label_encoder.fit_transform(df['CUST_GROUP'])
df['PRICE_GRP']= label_encoder.fit_transform(df['PRICE_GRP'])
df['ZCRMPRD']= label_encoder.fit_transform(df['ZCRMPRD'])
df['ZMLYGRP']= label_encoder.fit_transform(df['ZMLYGRP'])
df['ZPRDHYR4']= label_encoder.fit_transform(df['ZPRDHYR4'])
df['ZPRDHYR5']= label_encoder.fit_transform(df['ZPRDHYR5'])
df['ZPRDHYR6']= label_encoder.fit_transform(df['ZPRDHYR6'])
df['ZPRDHYR7']= label_encoder.fit_transform(df['ZPRDHYR7'])
df['ZPRDHYR8']= label_encoder.fit_transform(df['ZPRDHYR8'])
df['ZRPRGRP']= label_encoder.fit_transform(df['ZRPRGRP'])
df['ZSIKAYET']= label_encoder.fit_transform(df['ZSIKAYET'])
df['ZSIKAYET2']= label_encoder.fit_transform(df['ZSIKAYET2'])
df['ZSIKAYET3']= label_encoder.fit_transform(df['ZSIKAYET3'])
df['ZURNTIP']= label_encoder.fit_transform(df['ZURNTIP'])
df['ZZMARKA']= label_encoder.fit_transform(df['ZZMARKA'])
df['MATL_GROUP']= label_encoder.fit_transform(df['MATL_GROUP'])


df.head(5)
df.dtypes

# Mutual Information
# with that method we will see correlation between each features and target variable

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# how many input variable we want to remain
select_k = 18

selection = SelectKBest(mutual_info_classif, k=select_k).fit(df, target)
df.dtypes
# The number of variables to remain is indicated by select_k.
features = df.columns[selection.get_support()]
print(features)

#This variable has minimal correlation
min_cor_features = df.columns.difference(features)
min_cor_features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced,
                                                np.unique(y),
                                               y)

# ['balanced', 'calculated balanced', 'normalized'] are hyperparameters which we can play with.


#### Modeling #####

y=df["MATL_GROUP"]
y = y.astype("int64")
y.dtypes

type(X)
X= df.drop(["MATL_GROUP"],axis=1)
X.dtypes
X.head()

#### LIGHTGBM ####
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model
lgbm_params = {"learning_rate": [0.01, 0.02, 0.03, 0.1, 0.001],
               "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1],
               "n_estimators": [100, 250, 300, 350, 500, 1000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params,
                              cv=5, n_jobs=-1, verbose=True).fit(X,y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,
                                   random_state=17).fit(X,y)

cv_results =cross_validate(lgbm_final, X,y.values, cv=10,scoring=z,error_score="raise")
cv_results
type(y)
z = ['precision_macro', 'recall_macro',"accuracy","f1_macro"]


cv_results['test_accuracy'].mean()
cv_results['test_f1_macro'].mean()
cv_results["test_precision_macro"].mean()
cv_results['test_recall_macro'].mean()
cv_results["test_roc_auc"].mean() ## kullanamadım binary classificationda oluyormuş


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X)


##### CART ####

pip install pydotplus
pip install skompiler
skompiler: scikitlearn modellerini executable kodlara çevirir
pip install astor
pip install joblib

import warnings
import joblib
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)


cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
y_pred = cart_model.predict(X)


cv_results = cross_validate(cart_model,
                            X, y,
                            cv=10)

cv_results["test_score"].mean()
df.shape

#HYPERPARAMETER OPT.

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)


cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,
                                    random_state=17).fit(X, y)
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=10)
cart_best_grid.best_params_


cv_results["test_score"].mean()