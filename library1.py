import pandas as pd
import numpy as np

from scipy import interpolate

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
  

# 차분
class DiffSmooth(BaseEstimator, TransformerMixin) :
  def __init__(self, lags_n, diffs_n, smooth_n, diffs_abs = False, abs_features = False) :
    self.lags_n = lags_n
    self.diffs_n = diffs_n
    self.smooth_n = smooth_n
    self.diffs_abs = diffs_abs
    self.abs_features = abs_features

  def fit(self, X, y = None) :
    return self

  def transform(self, X) :
    df = X.copy()

    if self.diffs_n >= 1 :
      df = df.diff(self.diffs_n).dropna()
      if self.diffs_abs == False :
        df = abs(df)
    
    if self.smooth_n >= 2 :
      df = df.rolling(self.smooth_n).mean().dropna()

    if self.lags_n >= 1 :
      df_columns_new = [f'{col}_lag{n}' for n in range(self.lags_n + 1) for col in df.columns]
      df = pd.concat([df.shift(n) for n in range(self.lags_n + 1)], axis = 1).dropna()
      df.columns = df_columns_new

    df = df.reindex(sorted(df.columns), axis = 1)
    if self.abs_features == True :
      df = abs(df)
    
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
  



class PreprocessPipe(BaseEstimator, TransformerMixin) :
  def __init__(self) :
    preprocess_pipe = Pipeline([
      ('feature_selector', VolTempSelector('M')),
      ('missing_value', handleMissingValue()),
      ('siff_smooth_df', DiffSmooth(0, 0, 0)),
      ('pca', PCA(n_components = 3)),
      ('component_selector', ComponentSelector(3))
    ])

    self.pp = preprocess_pipe

  def fit(self, X, y = None) :
    return self.pp.fit(X)

  def transform(self, X) :
    return self.pp.transform(X)
