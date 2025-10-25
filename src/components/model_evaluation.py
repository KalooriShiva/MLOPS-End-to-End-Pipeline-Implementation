from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
import os
from typing import Optional
from dataclasses import dataclass
import json
import yaml
from datetime import datetime

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[object]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get the best model from the local artifact directory.
        
        Output      :   Returns a loaded model object if available, otherwise None.
        On Failure  :   Write an exception log and then raise an exception.
        """
        try:
            best_model_path = self.model_eval_config.best_model_path
            
            # Check if the model file exists locally
            if not os.path.exists(best_model_path):
                logging.info("No existing model found in local artifacts. This must be the first run.")
                return None
            
            # Load the model from the local file
            best_model = load_object(file_path=best_model_path)
            logging.info("Successfully loaded existing model from local artifacts.")
            return best_model
        except Exception as e:
            raise MyException(e, sys)
        
    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df
    

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function evaluates the newly trained model against the
                        existing model from the local artifacts directory.
        
        Output      :   Returns an EvaluateModelResponse object with comparison results.
        On Failure  :   Write an exception log and then raise an exception.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for the newly trained model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model() # This now gets the model from the local file system

            if best_model is not None:
                logging.info(f"Computing F1_Score for the existing production model...")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            
            # SAVE METRICS HERE
            self.save_evaluation_metrics(result)
            
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates all steps of the model evaluation.
        
        Output      :   Returns a model evaluation artifact.
        On Failure  :   Write an exception log and then raise an exception.
        """  
        try:
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
                # Add metrics paths
                evaluation_report_file_path=self.model_eval_config.evaluation_report_file_path,
                metrics_file_path=self.model_eval_config.metrics_file_path,
                trained_model_f1_score=evaluate_model_response.trained_model_f1_score,
                best_model_f1_score=evaluate_model_response.best_model_f1_score
            )
            
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def save_evaluation_metrics(self, evaluation_response: EvaluateModelResponse) -> None:
        """Save detailed evaluation metrics to artifacts"""
        try:
            # Create evaluation directory
            os.makedirs(os.path.dirname(self.model_eval_config.metrics_file_path), exist_ok=True)
            
            # Prepare metrics data
            metrics_data = {
                "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trained_model_f1_score": float(evaluation_response.trained_model_f1_score),
                "best_model_f1_score": float(evaluation_response.best_model_f1_score) if evaluation_response.best_model_f1_score else None,
                "performance_improvement": float(evaluation_response.difference),
                "is_model_accepted": evaluation_response.is_model_accepted,
                "change_threshold": self.model_eval_config.change_threshold,
                "trained_model_path": self.model_trainer_artifact.trained_model_file_path,
                "best_model_path": self.model_eval_config.best_model_path
            }
            
            # Save as JSON
            with open(self.model_eval_config.metrics_file_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            
            # Save summary as YAML
            summary_data = {
                "model_evaluation_summary": {
                    "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_acceptance": {
                        "accepted": evaluation_response.is_model_accepted,
                        "improvement": float(evaluation_response.difference),
                        "threshold": self.model_eval_config.change_threshold
                    },
                    "performance_metrics": {
                        "new_model_f1": float(evaluation_response.trained_model_f1_score),
                        "current_model_f1": float(evaluation_response.best_model_f1_score) if evaluation_response.best_model_f1_score else "No existing model"
                    }
                }
            }
            
            with open(self.model_eval_config.evaluation_report_file_path, 'w') as f:
                yaml.dump(summary_data, f, default_flow_style=False)
            
            logging.info(f"Evaluation metrics saved to: {self.model_eval_config.metrics_file_path}")
            logging.info(f"Evaluation report saved to: {self.model_eval_config.evaluation_report_file_path}")
            
        except Exception as e:
            raise MyException(e, sys)