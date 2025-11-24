from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import phik
from sklearn.preprocessing import RobustScaler


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in the dataframe for the specified features.

    Args:
        median_cols (list[str], optional): List of columns to impute by median. Defaults to None.
        zero_cols (list[str], optional): List of columns to impute with zero. Defaults to None.
        mode_cols (list[str], optional): List of columns to impute by mode. Defaults to None.
        add_missing_flags (bool, optional): If to add missingness indicators. Defaults to True.
    """

    def __init__(
        self,
        median_cols: list[str] = None,
        zero_cols: list[str] = None,
        mode_cols: list[str] = None,
        add_missing_flags: bool = True,
    ):
        self.median_cols = median_cols
        self.zero_cols = zero_cols
        self.mode_cols = mode_cols
        self.add_missing_flags = add_missing_flags

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CustomImputer":
        """
        Fits transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (CustomImputer): The fitted transformer.
        """
        self.fill_values_ = {}

        if self.median_cols is not None:
            for col in self.median_cols:
                self.fill_values_[col] = X[col].median()

        if self.zero_cols is not None:
            for col in self.zero_cols:
                self.fill_values_[col] = 0

        if self.mode_cols is not None:
            for col in self.mode_cols:
                self.fill_values_[col] = X[col].mode(dropna=True)[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe by imputing missing values.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        X = X.copy()

        for col, fill_value in self.fill_values_.items():
            new_cols = {}
            if self.add_missing_flags:
                new_cols[f"{col}_IS_MISSING"] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(fill_value)

            X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

        return X


class NewFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds additional features to the dataframe.

    Args:
        doc_threshold (float, optional): Threshold to evaluate feature relevance. Defaults to 0.02.
    """

    def __init__(self, doc_threshold: float = 0.02):
        self.doc_threshold = doc_threshold
        self.location_mismatch_cols_ = None
        self.bureau_cols_ = None
        self.common_flags_ = None
        self.rare_flags_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NewFeatureEngineer":
        """
        Fist the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (NewFeatureEngineer): The fitted transformer.
        """
        self.location_mismatch_cols_ = [
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]

        self.bureau_cols_ = [col for col in X.columns if "AMT_REQ_CREDIT_BUREAU" in col]
        flag_doc_cols = [col for col in X.columns if "FLAG_DOCUMENT" in col]
        threshold = self.doc_threshold * len(X)
        self.common_flags_ = [col for col in flag_doc_cols if X[col].sum() > threshold]
        self.rare_flags_ = [
            col for col in flag_doc_cols if col not in self.common_flags_
        ]

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
        new_cols = {}

        # Location Mismatch Feature
        if all(col in X.columns for col in self.location_mismatch_cols_):
            X["LOCATION_MISMATCH_COUNT"] = X[self.location_mismatch_cols_].sum(axis=1)
            X.drop(columns=self.location_mismatch_cols_, inplace=True)

        # Bureau Requests Features
        if len(self.bureau_cols_) > 0:
            new_cols["BUREAU_REQS_IS_MISSING"] = (
                X[self.bureau_cols_].isnull().any(axis=1).astype(int)
            )
            X[self.bureau_cols_] = X[self.bureau_cols_].fillna(0)
            X["BUREAU_REQS_TOTAL"] = X[self.bureau_cols_].sum(axis=1)
            X.drop(columns=self.bureau_cols_, inplace=True)

        # Rare Flag Document Features
        if len(self.rare_flags_) > 0:
            X["FLAG_DOCUMENT_RARE_COUNT"] = X[self.rare_flags_].sum(axis=1)
            X.drop(columns=self.rare_flags_, inplace=True)

        if "FLAG_OWN_REALTY" in X.columns:
            X["FLAG_OWN_REALTY"] = (
                X["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0}).astype(int)
            )

        # Car Ownership Feature
        if "OWN_CAR_AGE" in X.columns and "FLAG_OWN_CAR" in X.columns:
            X["FLAG_OWN_CAR"] = X["FLAG_OWN_CAR"].map({"Y": 1, "N": 0}).astype(int)

        X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]
        X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
        X["GOODS_PRICE_INCOME_RATIO"] = X["AMT_GOODS_PRICE"] / X["AMT_INCOME_TOTAL"]
        X["INCOME_PER_PERSON"] = X["AMT_INCOME_TOTAL"] / (X["CNT_FAM_MEMBERS"] + 1)
        X["INCOME_PER_CHILD"] = X["AMT_INCOME_TOTAL"] / (X["CNT_CHILDREN"] + 1)
        X["CREDIT_PER_PERSON"] = X["AMT_CREDIT"] / (X["CNT_FAM_MEMBERS"] + 1)
        X["CREDIT_PER_CHILD"] = X["AMT_CREDIT"] / (X["CNT_CHILDREN"] + 1)

        X["CREDIT_TERM"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]
        X["GOODS_PRICE_CREDIT_RATIO"] = X["AMT_GOODS_PRICE"] / X["AMT_CREDIT"]

        X["INCOME_PER_CREDIT"] = X["AMT_INCOME_TOTAL"] / X["AMT_CREDIT"]
        X["CAR_AGE_PER_INCOME"] = X["OWN_CAR_AGE"] / (X["AMT_INCOME_TOTAL"] + 1)

        X["EXT_SOURCE_SUM"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].sum(
            axis=1
        )
        X["EXT_SOURCE_1_RATIO"] = X["EXT_SOURCE_1"] / X["EXT_SOURCE_SUM"]
        X["EXT_SOURCE_2_RATIO"] = X["EXT_SOURCE_2"] / X["EXT_SOURCE_SUM"]
        X["EXT_SOURCE_3_RATIO"] = X["EXT_SOURCE_3"] / X["EXT_SOURCE_SUM"]
        X["EXT_SOURCE_PROD"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].prod(
            axis=1
        )
        X["EXT_SOURCE_MEAN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
            axis=1
        )
        X["EXT_SOURCE_STD"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(
            axis=1
        )
        X["EXT_SOURCE_MIN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(
            axis=1
        )
        X["EXT_SOURCE_MAX"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(
            axis=1
        )
        X["EXT_SOURCE_RANGE"] = X["EXT_SOURCE_MAX"] - X["EXT_SOURCE_MIN"]
        X["EXT_SOURCE_1_MISSING"] = X["EXT_SOURCE_1"].isnull().astype(int)
        X["EXT_SOURCE_2_MISSING"] = X["EXT_SOURCE_2"].isnull().astype(int)
        X["EXT_SOURCE_3_MISSING"] = X["EXT_SOURCE_3"].isnull().astype(int)
        X["EXT_SOURCE_W_ANNUITY"] = X["EXT_SOURCE_SUM"] * X["AMT_ANNUITY"]

        if "DAYS_EMPLOYED" in X.columns:
            X["IS_UNEMPLOYED"] = (X["DAYS_EMPLOYED"] == 365243).astype(int)
            X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)
            new_cols["DAYS_EMPLOYED_IS_MISSING"] = (
                X["DAYS_EMPLOYED"].isnull().astype(int)
            )
            X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].fillna(0)
            X["DAYS_EMPLOYED_PER_BIRTH"] = (X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]).abs()

        X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

        return X


class LowVarianceDropper(BaseEstimator, TransformerMixin):
    """
    Drops the features by variance threshold.

    Args:
        threshold (float, optional): Threshold for numerical features. Defaults to 0.01.
        cat_threshold (float, optional): Threshold for categorical features. Defaults to 0.99.
        verbose (bool, optional): If to be explicit about transformations. Defaults to True.
    """

    def __init__(
        self, threshold: float = 0.01, cat_threshold: float = 0.99, verbose: bool = True
    ):
        self.threshold = threshold
        self.cat_threshold = cat_threshold
        self.verbose = verbose
        self.num_features_to_keep_ = None
        self.cat_features_to_keep_ = None
        self.features_to_keep_ = None
        self.features_dropped_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "LowVarianceDropper":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (LowVarianceDropper): The fitted transformer instance.
        """
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        # --- Numeric ---
        if len(num_cols) > 0:
            scaler = RobustScaler()
            X_scaled_temp = scaler.fit_transform(X[num_cols].copy())
            selector = VarianceThreshold(threshold=self.threshold)
            selector.fit(X_scaled_temp)

            keep_num = num_cols[selector.get_support()]
            drop_num = num_cols.difference(keep_num)
        else:
            keep_num, drop_num = [], []

        # ------- Categorical ---
        if len(cat_cols) > 0:
            keep_cat, drop_cat = [], []
            for col in cat_cols:
                top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_freq >= self.cat_threshold:
                    drop_cat.append(col)
                else:
                    keep_cat.append(col)
        else:
            keep_cat, drop_cat = [], []

        self.num_features_to_keep_ = list(keep_num)
        self.cat_features_to_keep_ = list(keep_cat)
        self.features_to_keep_ = self.num_features_to_keep_ + self.cat_features_to_keep_
        self.features_dropped_ = list(drop_num) + list(drop_cat)

        if self.verbose and self.features_dropped_:
            print(
                f"""Dropping {len(self.features_dropped_)} low-variance features:
            {pd.Series(self.features_dropped_)}"""
            )

            print(f"Remaining features: {len(self.features_to_keep_)}.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input dataframe.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        return X[self.features_to_keep_].copy()


class PhikSelector(BaseEstimator, TransformerMixin):
    """
    Selects relevant predictors using Phi K correlation with the target.

    Args:
        relevance_threshold (float, optional): Correlation threshold for predictor-target correlation.
        Defaults to 0.05.
    """

    def __init__(self, relevance_threshold: float = 0.05):
        self.relevance_threshold = relevance_threshold
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PhikSelector":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series): Target series.

        Returns:
            self (PhikSelector): Fitted transformer.
        """
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy().rename("target")
        id_cols = ["SK_ID_CURR"]
        X = X.drop(columns=id_cols, errors="ignore")

        matrix = X.join(y)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        phik_matrix = phik.phik_matrix(matrix, interval_cols=numeric_cols)
        relevance = phik_matrix["target"].drop("target", errors="ignore")

        relevant_features = relevance[
            relevance >= self.relevance_threshold
        ].sort_values(ascending=False)

        if relevant_features.empty:
            self.selected_features_ = []
            print("No features meet the relevance threshold.")
            return self

        self.dropped_features_ = relevance[
            relevance < self.relevance_threshold
        ].index.tolist()
        initial_features = relevant_features.index.tolist()
        print(
            f"Selected {len(initial_features)} relevant features with phik >= {self.relevance_threshold}"
        )
        self.selected_features_ = initial_features
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the inout dataframe by selecting the relevant features.

        Args:
            X (pd.DataFrame): The input dataframe.

        Raises:
            ValueError: When selected features are not in the input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        missing_features = [
            feat for feat in self.selected_features_ if feat not in X.columns
        ]
        if missing_features:
            raise ValueError(
                f"The following selected features are not in the input data: {missing_features}"
            )

        return X[self.selected_features_].copy()


class FeatureInteractions(BaseEstimator, TransformerMixin):
    """
    Adds feature interactions.

    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureInteractions":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The predictors' matrix.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (FeatureInteractions): Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe by adding feature interactions.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
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
        X["EXT_SOURCE_MIN_INCOME_TYPE"] = X["EXT_SOURCE_MIN"] * X["NAME_INCOME_TYPE"]
        X["EXT_SOURCE_MAX_INCOME_TYPE"] = X["EXT_SOURCE_MAX"] * X["NAME_INCOME_TYPE"]

        X["EXT_SOURCE_3_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_3"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )

        X["EXT_SOURCE_MIN_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_MIN"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )

        X["EXT_SOURCE_MAX_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_MAX"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )

        X["EXT_SOURCE_1_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_1"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )

        X["EXT_SOURCE_2_MEAN_RATIO"] = np.where(
            X["EXT_SOURCE_MEAN"].notna(),
            X["EXT_SOURCE_2"] / X["EXT_SOURCE_MEAN"],
            np.nan,
        )

        X["MAX_AVG_UTILIZATION_EXT_SOURCE_MAX_RATIO"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["MAX_CC_AVG_UTILIZATION_RATIO"] / X["EXT_SOURCE_MAX"],
            np.nan,
        )

        X["MAX_MAX_UTILIZATION_EXT_SOURCE_MAX_RATIO"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["MAX_CC_MAX_UTILIZATION_RATIO"] / X["EXT_SOURCE_MAX"],
            np.nan,
        )

        X["AVG_DAYS_FIRST_DRAWING_EXT_SOURCE_MAX"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["AVG_DAYS_FIRST_DRAWING"] * X["EXT_SOURCE_MAX"],
            np.nan,
        )

        X["AVG_DAYS_FIRST_DRAWING_DAYS_LAST_OVERDUE_RATIO"] = np.where(
            X["BUREAU_DAYS_LAST_OVERDUE"].notna(),
            X["AVG_DAYS_FIRST_DRAWING"] / X["BUREAU_DAYS_LAST_OVERDUE"],
            np.nan,
        )

        X["CLOSED_CREDIT_EXT_SOURCE_MAX_RATIO"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["BUREAU_CLOSED_CREDIT_RATIO"] / X["EXT_SOURCE_MAX"],
            np.nan,
        )

        X["MAX_DOWNPAYMENT_TO_GOODS_EXT_SOURCE_MAX_RATIO"] = np.where(
            X["EXT_SOURCE_MAX"].notna(),
            X["MAX_DOWNPAYMENT_TO_GOODS_RATIO"] / X["EXT_SOURCE_MAX"],
            np.nan,
        )

        X = X.drop(columns=["FLOORSMAX_MODE", "FLOORSMAX_AVG"])

        return X


class FeatureInteractionsBoruta(BaseEstimator, TransformerMixin):
    """
    Adds feature interactions to the dataframe.

    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureInteractionsBoruta":
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): The original predictor dataframe.
            y (pd.Series, optional): The target series. Defaults to None.

        Returns:
            self (FeatureInteractionsBoruta): The fitted transformer instance.
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

        X = X.drop(columns=["FLAG_DOCUMENT_3", "FLOORSMAX_MODE", "FLOORSMAX_AVG"])
        return X
