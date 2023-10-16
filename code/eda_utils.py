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

def corr_colormap(df, annot=False):
    
    df = df.corr()
    
    sns.clustermap(df, 
               annot = annot,      # 실제 값 화면에 나타내기
               cmap = 'RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
               vmin = -1, vmax = 1, #컬러차트 -1 ~ 1 범위로 표시
              )
    

def x_correlation(train, LABEL):
    x_train = train.drop(columns=LABEL)
    corr_x = train.corr()[LABEL].drop(LABEL)

    plt.bar(x=x_train.columns, height=corr_x)
    plt.xticks(rotation=90, size=7)
    plt.show()
    
def avgOftime(df, col, LABEL="CI_HOUR", viz=False):
    
    avg_info = df.groupby(col).mean()[LABEL]
    col_info = df[col].unique()
    
    if viz:
        plt.figure(figsize=(20,8))
        plt.bar(col_info, avg_info.values)
        plt.xticks(col_info, rotation=90)
        plt.show()
    
def round_fn(number):
    return round(number, 1)
        
def info_per_col(df, categoric_features):
    
    columns = df.columns
    
    for col in columns:
        if col not in categoric_features and col not in ["ATA"]:
            print(col, "information")
            print("Min:", round_fn(df[col].min()),
                  "Max:", round_fn(df[col].max()),
                  "Mean", round_fn(df[col].mean()))
            plt.figure(figsize=(10,5))
            sns.boxplot(data=df[col])
            plt.title(f"{col} Feature")
            plt.show()
            
    