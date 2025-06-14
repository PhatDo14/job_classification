import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler , LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file = os.path.join('artifacts', "preprocessor.pkl")  # Where to save feature processor
    target_encoder_file = os.path.join('artifacts', "target_encoder.pkl")  # Where to save target converter

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Tool to process text and categories
            preprocessor = ColumnTransformer(transformers=[
                ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
                ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
                ("description_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95),
                 "description"),
                ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
                ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
            ])
            logging.info("Created tool to process data")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)  # Training data
            test_df = pd.read_csv(test_path)    # Test data
            logging.info(f"Loaded data: train {train_df.shape}, test {test_df.shape}")

            # Check if all columns exist
            needed_columns = ['title', 'location', 'description', 'function', 'industry', 'career_level']
            if not all(col in train_df.columns for col in needed_columns):
                raise CustomException(f"Missing columns in train data: {needed_columns}", sys)

            # Get tool to process features
            preprocessor = self.get_data_transformer_object()

            # Split features (everything except career_level) and target (career_level)
            target = "career_level"
            train_features = train_df.drop(columns=[target])
            train_target = train_df[target]
            test_features = test_df.drop(columns=[target])
            test_target = test_df[target]
            logging.info(f"Train features: {train_features.shape}, Train target: {train_target.shape}")

            # Convert career_level text to numbers (e.g., specialist -> 0)
            target_encoder = LabelEncoder()
            train_target_encoded = target_encoder.fit_transform(train_target)
            test_target_encoded = target_encoder.transform(test_target)
            logging.info("Converted career_level to numbers")

            # Save target converter
            save_object(self.config.target_encoder_file, target_encoder)
            logging.info(f"Saved target converter to {self.config.target_encoder_file}")

            # Process features
            train_features_arr = preprocessor.fit_transform(train_features)
            test_features_arr = preprocessor.transform(test_features)

            # Convert to normal array if needed (fixes sparse data)
            if hasattr(train_features_arr, 'toarray'):
                train_features_arr = train_features_arr.toarray()
                test_features_arr = test_features_arr.toarray()
            logging.info(f"Processed train features: {train_features_arr.shape}")

            # Check if sizes match
            if train_features_arr.shape[0] != train_target_encoded.shape[0]:
                raise CustomException(
                    f"Error: Features have {train_features_arr.shape[0]} rows, but target has {train_target_encoded.shape[0]} rows",
                    sys
                )

            # Combine features and target
            train_arr = np.c_[train_features_arr, train_target_encoded]
            test_arr = np.c_[test_features_arr, test_target_encoded]
            logging.info(f"Final data: train {train_arr.shape}, test {test_arr.shape}")

            # Save feature processor
            save_object(self.config.preprocessor_file, preprocessor)
            logging.info(f"Saved feature processor to {self.config.preprocessor_file}")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_file,
                self.config.target_encoder_file
            )

        except Exception as e:
            raise CustomException(e, sys)