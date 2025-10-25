from dataclasses import dataclass
from typing import Optional  # Import Optional for optional type hinting


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    trained_model_path: str
    changed_accuracy: float
    # Make these optional with default values
    evaluation_report_file_path: str = None
    metrics_file_path: str = None
    trained_model_f1_score: float = 0.0
    best_model_f1_score: Optional[float] = None

@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str