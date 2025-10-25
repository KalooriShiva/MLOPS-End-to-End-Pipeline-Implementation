import os
import sys
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException
from src.logger import logging

def run_model_trainer_from_artifacts(artifact_timestamp: str):
    """
    Run model trainer using existing transformation artifacts
    
    Args:
        artifact_timestamp: The timestamp folder of existing artifacts (e.g., "10_25_2025_19_16_56")
    """
    try:
        logging.info("Starting model trainer with existing artifacts...")
        
        # Path to existing transformation artifacts
        base_artifact_path = os.path.join("artifacts", artifact_timestamp)
        
        # Create DataTransformationArtifact pointing to existing files
        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=os.path.join(
                base_artifact_path, 
                "data_transformation", 
                "transformed_object", 
                "preprocessing.pkl"
            ),
            transformed_train_file_path=os.path.join(
                base_artifact_path,
                "data_transformation",
                "transformed",
                "train.npy"
            ),
            transformed_test_file_path=os.path.join(
                base_artifact_path,
                "data_transformation", 
                "transformed",
                "test.npy"
            )
        )
        
        # Verify all required files exist
        required_files = [
            data_transformation_artifact.transformed_object_file_path,
            data_transformation_artifact.transformed_train_file_path,
            data_transformation_artifact.transformed_test_file_path
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise Exception(f"Required artifact file not found: {file_path}")
        
        logging.info("All transformation artifacts found. Proceeding with model training...")
        
        # Initialize model trainer with config
        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config
        )
        
        # Run model training
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        logging.info("Model training completed successfully!")
        logging.info(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
        
        return model_trainer_artifact
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise MyException(e, sys)

if __name__ == "__main__":
    # Replace with your actual artifact timestamp
    ARTIFACT_TIMESTAMP = "10_25_2025_19_16_56"  # Change this to your timestamp
    
    try:
        model_artifact = run_model_trainer_from_artifacts(ARTIFACT_TIMESTAMP)
        print(f"✅ Model training successful!")
        print(f"📁 Model saved at: {model_artifact.trained_model_file_path}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        sys.exit(1)