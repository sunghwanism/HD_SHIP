
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

import random

def preprocessing(train, test):
    # datetime 컬럼 처리
    train['ATA'] = pd.to_datetime(train['ATA'])
    test['ATA'] = pd.to_datetime(test['ATA'])

    # datetime을 여러 파생 변수로 변환
    for df in [train, test]:
        df['year'] = df['ATA'].dt.year
        df['month'] = df['ATA'].dt.month
        df['day'] = df['ATA'].dt.day
        df['hour'] = df['ATA'].dt.hour
        df['minute'] = df['ATA'].dt.minute
        df['weekday'] = df['ATA'].dt.weekday

    # datetime 컬럼 제거
    train.drop(columns='ATA', inplace=True)
    test.drop(columns='ATA', inplace=True)

    # Categorical 컬럼 인코딩
    categorical_features = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']
    encoders = {}

    for feature in tqdm(categorical_features, desc="Encoding features"):
        le = LabelEncoder()
        train[feature] = le.fit_transform(train[feature].astype(str))
        le_classes_set = set(le.classes_)
        test[feature] = test[feature].map(lambda s: '-1' if s not in le_classes_set else s)
        le_classes = le.classes_.tolist()
        bisect.insort_left(le_classes, '-1')
        le.classes_ = np.array(le_classes)
        test[feature] = le.transform(test[feature].astype(str))
        encoders[feature] = le

    # 결측치 처리
    # train.fillna(train.mean(), inplace=True)
    # test.fillna(train.mean(), inplace=True)
    # train.dropna(axis=0, inplace=True)
    
    return train, test, categorical_features


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore