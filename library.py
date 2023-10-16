import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y=None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
    # Check if the target column exists in the DataFrame
    if self.target_column not in X.columns:
      error_message = f"\nError: {self.__class__.__name__} - The target column '{self.target_column}' does not exist in the DataFrame.\n"
      raise AssertionError(error_message)

    X_ = X.copy()
    # Perform one-hot encoding on the target column
    X_ = pd.get_dummies(X, columns=[self.target_column], dummy_na=self.dummy_na, drop_first=self.drop_first)

    return X_

  def fit_transform(self, X, y=None):
    # self.fit(X, y)
    result = self.transform(X)
    return result
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.boundaries = None

  def fit(self, X, y=None):
    assert isinstance(X, pd.DataFrame), "Input data must be a pandas DataFrame"
    assert self.target_column in X.columns, f'Misspelling "{self.target_column}".'

    mean = X[self.target_column].mean()
    std = X[self.target_column].std()

    lower_boundary = mean - 3 * std
    upper_boundary = mean + 3 * std

    self.boundaries = (lower_boundary, upper_boundary)
    return self

  def transform(self, X):
    assert self.boundaries is not None, f'"{self.__class__.__name__}": Missing fit.'
    lower_boundary, upper_boundary = self.boundaries
    X_ = X.copy()
    X_[self.target_column] = np.clip(X_[self.target_column], lower_boundary, upper_boundary)
    return X_.reset_index(drop=True)

  def fit_transform(self, X, y=None):
    self.fit(X)
    return self.transform(X)

class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer'], "Fence must be 'inner' or 'outer'"
        self.target_column = target_column
        self.fence = fence
        self.boundaries = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "Input data must be a pandas DataFrame"
        assert self.target_column in X.columns, f'Misspelling "{self.target_column}".'

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        if self.fence == 'outer':
          outer_low = q1 - 3.0 * iqr
          outer_high = q3 + 3.0 * iqr
          self.boundaries = (outer_low, outer_high)
        else:
          inner_low = q1 - 1.5 * iqr
          inner_high = q3 + 1.5 * iqr
          self.boundaries = (inner_low, inner_high)


        return self

    def transform(self, X):
        assert self.boundaries is not None, f'"{self.__class__.__name__}": Missing fit.'

        lower_boundary, upper_boundary = self.boundaries
        X_ = X.copy()
        X_[self.target_column] = X_[self.target_column].clip(lower=lower_boundary, upper=upper_boundary)

        return X_.reset_index(drop=True)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.column = column
    self.median_ = None
    self.iqr_ = None

  def fit(self, X, y=None):
    # Calculate the median and IQR for the specified column
    self.median_ = X[self.column].median()
    Q1 = X[self.column].quantile(0.25)
    Q3 = X[self.column].quantile(0.75)
    self.iqr_ = Q3 - Q1
    return self

  def transform(self, X):
    # Apply the Robust Transformer transformation to the specified column
    X_ = X.copy()
    X_[self.column] = (X_[self.column] - self.median_) / self.iqr_
    X_[self.column].fillna(0, inplace=True)  # Fill NaN values with 0
    return X_

  def fit_transform(self, X, y=None):
    # Fit and transform the specified column
    self.fit(X)
    return self.transform(X)
