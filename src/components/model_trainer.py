import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(random_state=2024),
                "Decision Tree": DecisionTreeClassifier(random_state=2024),
                # "Gradient Boosting": GradientBoostingClassifier(),
                # "Logistic Regression": LogisticRegression(),
                # "XGBRegressor": XGBClassifier(),
                # "CatBoosting Regressor": CatBoostClassifier(verbose=False),
                # "AdaBoost Regressor": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                     'max_depth':[None, 5, 10, 15],
                },
                "Random Forest": {
                    # 'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [2]#, 16, 32, 64, 128, 256]
                }

            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.52:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            recall = recall_score(y_test, predicted, average='weighted')
            return recall,best_model_name,predicted


        except Exception as e:
            raise CustomException(e, sys)