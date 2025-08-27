import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.utils.exception import CustomException

logger = setup_logger()

@dataclass
class DataIngestionConfig:
    """Configuration for paths used in data ingestion."""
    train_data_path :str = os.path.join("data",'train.csv')
    test_data_path: str = os.path.join("data", 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')
    
class DataIngestion:
    """Handles actual process of ingesting raw data and spliting it into train/test sets."""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """Reads raw data, saves it, and splits it into training and testing sets."""
        logger.info("Starting Data Ingestion Process...")
        try:
            df = pd.read_csv(r'fetch/data/finviz.csv')
            logger.info("Successfully loaded dataset. Shape: %s", df.shape)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info("Raw data saved at %s", self.ingestion_config.raw_data_path)
            
            logger.info("Splitting dataset into train and test sets..") 
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info('Training data saved at %s', self.ingestion_config.train_data_path)
            logger.info('Testing data saved at %s', self.ingestion_config.test_data_path)
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                     
            
            
        except Exception as e:
            logger.error("Data Ingestion Failed: %s", str(e))
            raise CustomException(e,sys) from e
        
