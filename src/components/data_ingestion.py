import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import re
from imblearn.over_sampling import RandomOverSampler, SMOTEN

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

def filter_location(location):
    result1 = re.findall("\,\s[A-Z]{2}$", location)
    if len(result1) !=0 :
        return result1[0][2:]
    else:
        return location

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('data/job_dataset.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save data in the raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            df = df.dropna(axis=0)
            logging.info("DropNa")

            df["location"] = df["location"].apply(filter_location)
            logging.info("Feature Engineering for location feature")

            label = "career_level"
            x = df.drop(label, axis=1)
            y = df[label]

            # Train-test split with stratification
            logging.info("Train test split initiated")
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )

            # Apply SMOTEN to training data
            logging.info("Applying SMOTEN to training data")
            ros = SMOTEN(
                random_state=0,
                k_neighbors=2,
                sampling_strategy={
                    "director_business_unit_leader": 500,
                    "specialist": 500,
                    "managing_director_small_medium_company": 500,
                    "bereichsleiter": 1000
                }
            )
            x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

            # Combine resampled training data into train_set
            train_set = pd.concat([x_train_resampled, y_train_resampled], axis=1)
            # Combine test data into test_set
            test_set = pd.concat([x_test, y_test], axis=1)

            #save train_data in the train_data_path
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)

            #save test_data in the test_data_path
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)

            logging.info("Ingestion of data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    # obj.initiate_data_ingestion()
    train_data,test_data = obj.initiate_data_ingestion()
    #
    data_transformation = DataTransformation()
    # data_transformation.initiate_data_transformation(train_data, test_data)
    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    #
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
    # _, _, res= modeltrainer.initiate_model_trainer(train_arr, test_arr)
    # for i in range(len(res)):
    #     print(res[i])
    # print(res)