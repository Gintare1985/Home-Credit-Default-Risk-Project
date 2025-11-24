from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class NewFeatureEngineerDeploy(BaseEstimator, TransformerMixin):
    """
    Transformer to add feature interactions to the pipeline.

    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NewFeatureEngineerDeploy":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Predictors' matrix.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            self (NewFeatureEngineerDeploy): Returns the instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input dataframe by adding feature interactions.

        Args:
            X (pd.DataFrame): Inout dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        X = X.copy()

        X["CREDIT_TERM"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
        X["GOODS_PRICE_CREDIT_RATIO"] = X["AMT_GOODS_PRICE"] / X["AMT_CREDIT"]
        X["CREDIT_PER_PERSON"] = X["AMT_CREDIT"] / (X["CNT_FAM_MEMBERS"] + 1)

        X["EXT_SOURCE_SUM"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].sum(
            axis=1
        )
        X["EXT_SOURCE_1_RATIO"] = X["EXT_SOURCE_1"] / X["EXT_SOURCE_SUM"]

        X["EXT_SOURCE_PROD"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].prod(
            axis=1
        )
        X["EXT_SOURCE_MEAN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
            axis=1
        )
        X["EXT_SOURCE_MIN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(
            axis=1
        )
        X["EXT_SOURCE_MAX"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(
            axis=1
        ) 
        return X


class CategoricalAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates categorical features to broader groups.

    Args:
        mappings (dict[dict]): Mappings of categorical feature groups.
    """

    def __init__(self, mappings: dict[dict]):
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalAggregator":
        """
        Fits transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            self (CategoricalAggregator): Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Groups categorical features to broader groups.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        X = X.copy()
        for col, mapping in self.mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna("Other")
                X[col] = X[col].astype("category")
        return X


class DaysToYearsTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms days to years.

    Args:
        columns (list[str]): List of features to transform.
    """

    def __init__(self, columns: list[str] = None):
        self.columns = columns
        self.time_cols_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DaysToYearsTransformer":
        """
        _summary_

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (DaysToYearsTransformer): Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the original dataframe.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()
        if self.columns is not None:
            time_cols = self.columns
        else:
            time_cols = [col for col in X.columns if col.startswith("DAYS_")]

        for col in time_cols:
            if col in X.columns:
                years_col = -X[col] / 365
                clipped_years_col = years_col.clip(lower=0).abs()
                X[col.replace("DAYS", "YEARS")] = clipped_years_col
                X.drop(columns=[col], inplace=True)
        return X


class NumericDowncaster(BaseEstimator, TransformerMixin):
    """
    Downcasts numerical features for efficiency.

    Args:
        verbose (bool): If to explicit after transformations.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.downast_mapping_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NumericDowncaster":
        """
        Fits the trasnformer to the data.

        Args:
            X (pd.DataFrame): Predictors' matrix.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            self (NumericDowncaster): Fitted transformer.
        """
        self.bool_cols_ = X.select_dtypes(include=["bool"]).columns.tolist()
        for col in self.bool_cols_:
            self.downast_mapping_[col] = np.int8
            if self.verbose:
                print(f"Column '{col}' downcasted from bool to int8")

        for col in X.select_dtypes(include=[np.number]).columns:
            original_dtype = X[col].dtype

            if pd.api.types.is_integer_dtype(X[col]):
                downcasted = pd.to_numeric(X[col], downcast="integer")
            else:
                downcasted = pd.to_numeric(X[col], downcast="float")

            if self.verbose and downcasted.dtype != original_dtype:
                print(
                    f"Column '{col}' downcasted from {original_dtype} to {downcasted.dtype}"
                )

            self.downast_mapping_[col] = downcasted.dtype
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe.

         Args:
             X (pd.DataFrame): The original dataframe.

         Returns:
             pd.DataFrame:The transformed dataframe.
        """
        X = X.copy()

        for col, dtype in self.downast_mapping_.items():
            if col in X.columns:
                X[col] = X[col].astype(dtype)
        return X

    def get_dtypes(self) -> dict:
        """
        Returns the dictionary of column and data type pairs.

        Returns:
            dict: Dictionary of column - data type pairs.
        """
        return self.downast_mapping_


class CategoricalConverter(BaseEstimator, TransformerMixin):
    """
    Converts object data type features to category data type.

    Args:
        max_unique (int): Integer for maximum unique values of the object dtype feature.
        verbose (bool): To set explicit mode of the transformer.
    """

    def __init__(self, max_unique: int = 20, verbose: bool = True):
        self.max_unique = max_unique
        self.verbose = verbose
        self.to_convert_ = []
        self.not_converted_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalConverter":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (CategoricalConverter): Fitted transformer.
        """
        for col in X.select_dtypes(include=["object"]).columns:
            unique_count = X[col].nunique(dropna=False)
            if unique_count <= self.max_unique:
                self.to_convert_.append(col)
                if self.verbose:
                    print(
                        f"Column '{col}' will be converted to 'category' (unique values: {unique_count})"
                    )
            else:
                self.not_converted_.append(col)
                if self.verbose:
                    print(
                        f"Column '{col}' has too many unique values ({unique_count}), not converting."
                    )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the original dataframe.

        Args:
            X (pd.DataFrame): The inout dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()
        for col in self.to_convert_:
            X[col] = X[col].astype("category")
        return X

    def get_conversion_info(self) -> dict[list]:
        """
        Returns information about transformer output.

        Returns:
            dict[list]: Dictionary oof transformed and not feature lists.
        """
        return {
            "converted_columns": self.to_convert_,
            "not_converted_columns": self.not_converted_,
        }


class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Drops features from the dataframe

    Args:
        features_to_drop (list[str]): List of features to drop.
    """

    def __init__(self, features_to_drop: list[str] = None):
        self.features_to_drop = features_to_drop or []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureDropper":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series, optional): The target feature. Defaults to None.

        Returns:
            self (FeatureDropper): The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by dropping the specified features.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()
        return X.drop(columns=self.features_to_drop, errors="ignore")


class OutlierSkewTransformer(BaseEstimator, TransformerMixin):
    """
    Handles outliers by clipping features to the specified percentiles and log-transforming
    features with skewness greater than 1.

    Args:
        lower_pct (float, optional): Lower percentile to clip. Defaults to 1.0.
        upper_pct (float, optional): Upper percentile to clip. Defaults to 99.0.
        columns (list[str], optional): Specified columns to transform. Defaults to None.
        skew_threshold (float, optional): Threshold to evaluate extreme skewness. Defaults to 1.0.
        log_transform (bool, optional): If to apply log-transformation to the features. Defaults to True.
        verbose (bool, optional): If to be explicit about the results of the transformation. Defaults to True.
    """

    def __init__(
        self,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
        columns: list[str] = None,
        skew_threshold: float = 1.0,
        log_transform: bool = True,
        verbose: bool = True,
    ):

        self.columns = columns
        self.skew_threshold = skew_threshold
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.skewed_features_ = []
        self.clipped_features_ = []
        self.log_transformed_ = []
        self.percentiles_ = {}
        self.log_transform = log_transform
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "OutlierSkewTransformer":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            self (OutlierSkewTransformer): Fitted transformer.
        """
        if self.columns is not None:
            self.feature_names_ = self.columns
        else:
            self.feature_names_ = X.select_dtypes(include=[np.number]).columns

        for col in self.feature_names_:
            lower = np.percentile(X[col].dropna(), self.lower_pct)
            upper = np.percentile(X[col].dropna(), self.upper_pct)
            self.percentiles_[col] = (lower, upper)

            skewness = X[col].skew()
            if abs(skewness) > self.skew_threshold:
                self.skewed_features_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe.

        Args:
            X (pd.DataFrame): The input dataframe.

        Raises:
            ValueError: When defined column is not found in the input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()

        for col in self.feature_names_:
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")

        for col, (low, high) in self.percentiles_.items():
            before = X[col].copy()
            X[col] = np.clip(X[col], a_min=low, a_max=high)
            if not before.equals(X[col]):
                self.clipped_features_.append(col)

            if self.log_transform and col in self.skewed_features_:
                X[col] = np.log1p(X[col])
                self.log_transformed_.append(col)

        if self.verbose:
            print(
                f"Feature winsorization completed: {len(self.clipped_features_)} features clipped."
            )
            if self.log_transform:
                print(
                    f"Skewness handling completed: {len(self.log_transformed_)} features log-transformed."
                )
        return X

    def get_feature_info(self) -> dict[list]:
        """
        Returns the results of transformed features.

        Returns:
            dict[list]: List of transformed features.
        """
        return {
            "clipped_features": list(self.clipped_features_),
            "log_transformed": list(self.log_transformed_),
            "skewed_features": list(self.skewed_features_),
        }

    def set_output(self, *, transform: str = None) -> "OutlierSkewTransformer":
        """
        Sets the output configuration for the transformer.

        Args:
            transform (str, optional): Output container configuration
            requested by scikit-learn. Defaults to None.

        Returns:
            self (OutlierSkewTransformer): The transformer instance.
        """
        return self


class TableMerger(BaseEstimator, TransformerMixin):
    """
    Merges separate dataframes into one.

    Args:
        tables (dict): A dictionary of tables to merge.
        key (str, optional): Field to merge on. Defaults to 'SK_ID_CURR'.
        how (str, optional): Mode to merge tables. Defaults to 'left'.
        verbose (bool, optional): Explicitness mode. Defaults to False.
    """

    def __init__(
        self,
        tables: dict,
        key: str = "SK_ID_CURR",
        how: str = "left",
        verbose: bool = False,
    ):
        self.tables = tables
        self.key = key
        self.how = how
        self.verbose = verbose
        self.merged_columns_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "TableMerger":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series, optional): Target series. Defaults to None.

        Raises:
            ValueError: When key not found in the table.

        Returns:
            self (TableMerger): The fitted transformer.
        """
        for name, table in self.tables.items():
            if self.key not in table.columns:
                raise ValueError(f"Join key '{self.key}' not found in table '{name}'")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input dataframe by merging additional tables.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()
        original_index = X.index.copy()

        for name, table in self.tables.items():
            before_cols = set(X.columns)
            X = X.merge(table, on=self.key, how=self.how, sort=False)
            X.index = original_index
            new_cols = set(X.columns) - before_cols
            self.merged_columns_[name] = list(new_cols)
            if self.verbose:
                print(f"Merged table '{name}' with {len(new_cols)} new columns.")
        return X


class FeatureInteractionsBorutaDeploy(BaseEstimator, TransformerMixin):
    """
    Makes feature interactions.

    """

    def __init__(self):
        pass

    def fit(
        self, X: pd.DataFrame, y: pd.Series = None
    ) -> "FeatureInteractionsBorutaDeploy":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors; matrix.
            y (pd.Series, optional): Target series. Defaults to None.

        Returns:
            self (FeatureInteractionsBoruta): The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input dataframe.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()

        X["DOC_3_INCOME_TYPE_CONTRACT_STATUS"] = (
            X["FLAG_DOCUMENT_3"]
            * X["NAME_INCOME_TYPE"]
            * X["MODE_NAME_CONTRACT_STATUS"]
        )
        X["ORGANIZATION_OCCUPATION_TYPES"] = (
            X["ORGANIZATION_TYPE"] * X["OCCUPATION_TYPE"]
        )
        X["EXT_SOURCE_1_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_1"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )
        X["MAX_MAX_UTILIZATION_EXT_SOURCE_MAX_RATIO"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["MAX_CC_MAX_UTILIZATION_RATIO"] / X["EXT_SOURCE_MAX"],
            np.nan,
        )
        X["MAX_AVG_UTILIZATION_EXT_SOURCE_MAX_RATIO"] = (
            X["MAX_CC_AVG_UTILIZATION_RATIO"] / X["EXT_SOURCE_MAX"]
        )
        X["AVG_DAYS_FIRST_DRAWING_EXT_SOURCE_MAX"] = (
            X["AVG_DAYS_FIRST_DRAWING"] * X["EXT_SOURCE_MAX"]
        )

        X = X.drop(columns=["FLAG_DOCUMENT_3"])
        return X
