import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('split train and test input data')
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            
            models = {
                'AdaBoost Regressor': AdaBoostRegressor(),
                'GradientBoosting Regressor': GradientBoostingRegressor(),
                'RandomForest Regressor': RandomForestRegressor(),
                'Linear Regression': LinearRegression(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'DecisionTree Regressor': DecisionTreeRegressor(),
                'XGBRF Regressor': XGBRFRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }
            
            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            
            # Best model selection
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("no best model found")
            
            logging.info(f"Best model: {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            
            return r2_score_value
            
        except Exception as e:
            raise CustomException(e, sys)