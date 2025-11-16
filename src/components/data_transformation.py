import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    encoder_obj_file_path: str = os.path.join("artifacts", "encoder.pkl")
    schema_file_path: str = os.path.join("artifacts", "schema.json")
    feature_columns_file_path: str = os.path.join(
        "artifacts", "feature_columns.json"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = "fraud"
        self.numerical_columns = [
            "amount",
            "customer_age",
            "minute_of_day",
            "to_acc_volume",
            "session_duration",
        ]
        self.categorical_columns = ["hour_of_day", "day_of_week"]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Fill missing historical payment dates and convert to datetime
        for col in ["lut_first_paid_date", "lut_last_paid_date"]:
            if col in df.columns:
                df[col] = df[col].fillna("1970-01-01")

        date_cols = [
            "date",
            "loginTime",
            "lut_first_paid_date",
            "lut_last_paid_date",
            "txn_timestamp",
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Derive engineered temporal features
        if "txn_timestamp" in df.columns:
            df["hour_of_day"] = df["txn_timestamp"].dt.hour.fillna(0).astype(int)
            df["day_of_week"] = df["txn_timestamp"].dt.dayofweek.fillna(0).astype(int)
        else:
            df["hour_of_day"] = 0
            df["day_of_week"] = 0

        if {"txn_timestamp", "loginTime"}.issubset(df.columns):
            df["session_duration"] = (
                df["txn_timestamp"] - df["loginTime"]
            ).dt.total_seconds()
        else:
            df["session_duration"] = 0

        df["session_duration"] = df["session_duration"].fillna(0)

        # Ensure numeric columns are numeric
        for col in [
            "amount",
            "customer_age",
            "minute_of_day",
            "to_acc_volume",
            "session_duration",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df[self.target_column_name] = (
            pd.to_numeric(df[self.target_column_name], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        return df

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = self._engineer_features(pd.read_csv(train_path))
            test_df = self._engineer_features(pd.read_csv(test_path))

            logging.info("Read train and test data completed")

            target_column_name = self.target_column_name
            feature_columns = self.numerical_columns + self.categorical_columns

            missing_cols = [col for col in feature_columns if col not in train_df.columns]
            if missing_cols:
                raise CustomException(
                    f"Missing required columns for transformation: {missing_cols}", sys
                )

            input_feature_train_df = train_df[feature_columns].copy()
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns].copy()
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying StandardScaler and OneHotEncoder as per notebook.")

            imputer = SimpleImputer(strategy="median")
            input_feature_train_df[self.numerical_columns] = imputer.fit_transform(
                input_feature_train_df[self.numerical_columns]
            )
            input_feature_test_df[self.numerical_columns] = imputer.transform(
                input_feature_test_df[self.numerical_columns]
            )

            scaler = StandardScaler()
            train_num = scaler.fit_transform(
                input_feature_train_df[self.numerical_columns]
            )
            test_num = scaler.transform(input_feature_test_df[self.numerical_columns])

            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_cat = encoder.fit_transform(
                input_feature_train_df[self.categorical_columns]
            )
            test_cat = encoder.transform(input_feature_test_df[self.categorical_columns])

            input_feature_train_arr = np.hstack([train_num, train_cat])
            input_feature_test_arr = np.hstack([test_num, test_cat])

            feature_names = list(self.numerical_columns) + list(
                encoder.get_feature_names_out(self.categorical_columns)
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving scaler, encoder, and schema artifacts.")

            os.makedirs(os.path.dirname(self.data_transformation_config.schema_file_path), exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler,
            )

            save_object(
                file_path=self.data_transformation_config.encoder_obj_file_path,
                obj=encoder,
            )

            schema = {
                "num_cols": self.numerical_columns,
                "all_cols": feature_columns,
            }
            with open(self.data_transformation_config.schema_file_path, "w") as f:
                json.dump(schema, f)

            with open(
                self.data_transformation_config.feature_columns_file_path, "w"
            ) as f:
                json.dump(feature_names, f)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
