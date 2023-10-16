import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import bisect
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import seaborn as sns


CATEGORY_COL = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']



# Time Column Split
def timeTransformer(org_train, org_test, day_split=True, weekday_split=True, covid_year=True):
        
    train = org_train.copy()
    test = org_test.copy()
    
    train['ATA'] = pd.to_datetime(train['ATA'])
    test['ATA'] = pd.to_datetime(test['ATA'])

    for df in [train, test]:
        df['year'] = df['ATA'].dt.year
        df['month'] = df['ATA'].dt.month
        df['day'] = df['ATA'].dt.day
        df['hour'] = df['ATA'].dt.hour
        df['minute'] = df['ATA'].dt.minute
        df['weekday'] = df['ATA'].dt.weekday
        
        df.drop(columns="minute", inplace=True)
    
    # 시간을 dawn (0), morning(1), afternoon(2), evening(3)
    if day_split:
        for df in [train, test]:
            df["day_catg"] = -1
            df.loc[df['hour'] < 7, 'day_catg']  = 0
            df.loc[(7 <= df['hour']) & (df['hour'] < 12), 'day_catg']  = 1
            df.loc[(12 <= df['hour']) & (df['hour'] < 18), 'day_catg']  = 2
            df.loc[18 <= df['hour'], 'day_catg']  = 3
        CATEGORY_COL.append("day_catg")
            # df.drop(columns="hour", inplace=True)
            
    if weekday_split: # 0 = NOT weekend , 1 = weekend
        for df in [train, test]:
            df["weekend"] = 0
            df.loc[df['weekday']==5, 'weekend']  = 1
            df.loc[df['weekday']==6, 'weekend']  = 1
        CATEGORY_COL.append("weekday")
            
            
    if covid_year:
        for df in [train, test]:
            df["covid"] = 0
            df.loc[df['year'] >= 2019, 'covid']  = 1
        CATEGORY_COL.append("covid")
            
    # datetime 컬럼 제거
    train.drop(columns=['ATA', 'ATA_LT'], inplace=True)
    test.drop(columns=['ATA', 'ATA_LT'], inplace=True)
    
    return train, test

def BN_preprocessing(train, test, BN_split=True):
    """
    fill N/A -> 2018년 12월부터는 데이터가 존재
    1) 도착항의 국가 및 항구명이 동일하면서
    2) month가 동일한 데이터로 null값을 채움
    
    만약 2번의 조건에 모두 만족하지 않는다면, 1번 조건에 만족하는 값들의 mean값으로 null값을 대체
    만약 1번, 2번 조건 모두 만족하지 않는다면, ....
    """

    # Processing for Null values
    for df in [train, test]:
        
        NaN_count = 0
        one_interpolate = 0
        
        bn_null_df = df[df["BN"].isna()][["ARI_CO", "ARI_PO", "BN", "month"]]
        bn_null_idx = bn_null_df.index
        
        for i in tqdm(bn_null_idx):
            CO, PO, _, month = bn_null_df.iloc[i]
            same_df_1 = train.loc[(train["ARI_CO"]==CO)&(train["ARI_PO"]==PO)&(train["month"]==month)&(train["BN"].notna())]
            same_df_2 = test.loc[(test["ARI_CO"]==CO)&(test["ARI_PO"]==PO)&(test["month"]==month)&(test["BN"].notna())]
            
            if (len(same_df_1)+len(same_df_2)) == 0:
                interpolate = np.nan()
                NaN_count += 1
                
            elif (len(same_df_1)+len(same_df_2)) == 1:
                interpolate = same_df_1["BN"].mean() + same_df_2["BN"].mean()
                one_interpolate += 1
                
            else:
                interpolate = interpolate = same_df_1["BN"].mean() + same_df_2["BN"].mean()
                
            df.iloc[i, np.where(train.columns == "BN")[0]] = interpolate
        
        print(f"NaN data {NaN_count} // One interpolate {one_interpolate}")
    
    if BN_split:
        train["BN"] = train["BN"].round(0)
        test["BN"] = test["BN"].round(0)
        CATEGORY_COL.append("BN")
    
    train.dropna(axis=0, subset=["BN"])
    
    return train, test
    


def preprocessing(orgin_train, origin_test,
                  day_split=True, weekday_split=True, covid_year=True, # About Time
                  BN_cleaning=True, # make BN as categorical feature
                  ):
    
    print("[Time Transformer]")
    train, test = timeTransformer(orgin_train, origin_test,
                                  day_split=day_split, weekday_split=weekday_split, 
                                  covid_year=covid_year)
    print("Category feature after Time Transformer", CATEGORY_COL)
    print("------------------------------------------------")
    
    print("[BN Preprocessor]")
    train, test = BN_preprocessing(train, test, BN_split=BN_cleaning)
    print("Category feature after BN Preprocessor", CATEGORY_COL)
    print("------------------------------------------------")
    
    
    return train, test
    