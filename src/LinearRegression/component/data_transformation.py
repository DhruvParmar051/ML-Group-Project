import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.exception import CustomException
from src.utils.logger import setup_logger
from src.utils.save_object import save_object

logger = setup_logger()

@dataclass 
class DataTransformationConfig:
    """Configuration paths for data transformation artifacts"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Handles the actual Data Transformation"""
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_pipeline(self):
        """
        Create and returns a preprocessing pipeline that imputes missing values and scales numerical values
        """
        
        try:
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            logger.info('Numerical Pipeline Created')
            return num_pipeline

        except Exception as e:
            logger.info('Data Preprocessor Failed.. %s', str(e))
            raise CustomException(e, sys) from e
        

    def initiate_data_transformation(self, train_path: str, test_path: str, target_column: str):
        """
        Applied data transformation onto training and test dataset.
        """
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        
            
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)

            X_train = train_df[numeric_cols]
            y_train = train_df[target_column]
            
            y_train = np.log1p(y_train)
            
            X_test = test_df[numeric_cols]
            y_test = test_df[target_column]
            y_test  = np.log1p(y_test)
            
            Pipeline = self.get_data_transformation_pipeline()
            
            logger.info('Applying Pipeline')
            
            X_train_transformed = Pipeline.fit_transform(X_train)
            X_test_transformed = Pipeline.fit_transform(X_test)
            
            X_train_transformed = pd.DataFrame(
                X_train_transformed, columns=numeric_cols, index=train_df.index
            )
            X_test_transformed = pd.DataFrame(
                X_test_transformed, columns=numeric_cols, index=test_df.index
            )
            
            train_transformed = pd.concat([X_train_transformed, y_train], axis=1)
            test_transformed = pd.concat([X_test_transformed, y_test], axis=1)
            
            logger.info("💾 Saving preprocessing pipeline object...")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=Pipeline
            )
            
            return train_transformed, test_transformed, self.data_transformation_config.preprocessor_obj_file_path
                  
                  
            
        except Exception as e:
            logger.error("Data Transformation Failed %s", str(e))
            raise CustomException(e, sys) from e
        