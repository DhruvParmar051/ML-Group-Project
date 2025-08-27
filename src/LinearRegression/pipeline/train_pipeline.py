import sys

from src.LinearRegression.component.data_ingestion import DataIngestion
from src.LinearRegression.component.data_transformation import DataTransformation
from src.LinearRegression.component.model import ModelTrainer
from src.utils.logger import setup_logger
from src.utils.exception import CustomException


logger = setup_logger()

def run_pipeline():
    """Running Pipeline"""
    try:
        print("heelo")
        logger.info("Starting Finviz data analyzes Pipeline")
        logger.info("Initiating Data Ingestion...")
        
        print("heelo")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        logger.info("Data Ingestion Complete: Train: %s, Test: %s", train_path, test_path)
        logger.info("🔄 Initiating Data Transformation...")
        transformation = DataTransformation()
        train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path, 'Price'
        )
        logger.info("Data Transformation Complete. Preprocessor saved at: %s", preprocessor_path)

        logger.info("Initiating Model Training...")
        trainer = ModelTrainer()
        results = trainer.initiate_model_trainer(train_array, test_array)

        logger.info("Pipeline Execution Completed Successfully!")
        logger.info("Final Results: %s", results)

        return results
        
        
    except Exception as e:
        logger.error('Error occured during pipeline executing...')
        raise CustomException(e, sys) from e
    
if __name__ == '__main__':
    run_pipeline()