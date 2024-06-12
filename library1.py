import pandas as pd
import numpy as np

from scipy import interpolate

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin


################################## 전처리 ##################################

# 셀 전압/온도 변수만 추출하여 사용
class VolTempSelector(BaseEstimator, TransformerMixin) :
  def __init__(self, start_name) :
    self.start_name = start_name

  def fit(self, X, y = None) :
    return self

  def transform(self, X) :
    df = X.copy()
    return df.filter(regex = self.start_name)
  

# 결측치 처리
class handleMissingValue(BaseEstimator, TransformerMixin) :

  def fit(self, X, y = None) :
    return self

  def transform(self, X) :
    df = X.copy()
    df_null = df.isnull()  # 결측치 여부 확인
    r, c = np.where(df_null)  # 결측치가 존재하는 (행, 열) 인덱스 확인

    for i in range(len(r)) :
      # 결측치가 첫 번째 값인 경우
      if r[i] == 0 :
        s = df.iloc[:, c[i]]
        df.iloc[r[i], c[i]] = df.iloc[s.notna().idxmax(), c[i]]
      # 그 외 : interpolate 사용(선형보간법)
      else :
        df = df.interpolate()

    return df
  

# PCA 후 n개의 주성분만 선택
class ComponentSelector(BaseEstimator, TransformerMixin) :
  def __init__(self, n) :
    self.n = n

  def fit(self, X , y = None) :
    return self

  def transform(self, X) :
    df = pd.DataFrame(X)
    return df.iloc[:, 0:self.n]
  


################################## 모델링 ##################################

class pipeline_model :
  def __init__(self, params = None) :
    
    # 전처리 파이프라인
    preprocess_pipe = Pipeline([
        ('feature_selector', VolTempSelector('M')),
        ('missing_value', handleMissingValue()),
        ('pca', PCA(n_components = 3)),
        ('minmax_scaler', MinMaxScaler()),
        ('component_selector', ComponentSelector(3))
        ])