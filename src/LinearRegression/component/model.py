import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.utils.exception import CustomException
from src.utils.logger import setup_logger
from src.utils.save_object import save_object

logger = setup_logger()


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """Class responsible for training and saving the best model."""

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train Linear Regression Model on training data, evaluate on test data,
        save model, and return performance metrics.
        """
        try:
            logger.info("📊 Splitting training and testing data...")

            X_train, y_train, X_test, y_test = (
                train_array.iloc[:, :-1],
                train_array.iloc[:, -1],
                test_array.iloc[:, :-1],
                test_array.iloc[:, -1],
            )

            logger.info("✅ Initialized Linear Regression Model")

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            logger.info("📈 Model Evaluation Metrics - R2: %.4f, RMSE: %.4f, MAE: %.4f", r2, rmse, mae)

            logger.info("💾 Saving trained model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logger.info("✅ Model Training Completed Successfully!")

            # return both trained model and metrics
            return {
                "model_path": self.model_trainer_config.trained_model_file_path,
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae
            }

        except Exception as e:
            logger.error("❌ Error occurred during Model Training")
            raise CustomException(e, sys) from e
