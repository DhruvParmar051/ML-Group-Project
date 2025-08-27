import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.exception import CustomException
from utils.logger import setup_logger

logger = setup_logger()

@dataclass 
class DataTransformationConfig:
    """Configuation paths for data transformation artifacrs"""
    preproceesor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    

class DataTransformation:
    """Handles the actual Data Transformation"""
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self, df:pd.DataFrame):
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
            logger.info()