import sys
from src.entity.config_entity import VehiclePredictorConfig
# The S3 estimator is no longer needed
# from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
from src.utils.main_utils import load_object


class VehicleData:
    def __init__(self,
                Gender,
                Age,
                Driving_License,
                Region_Code,
                Previously_Insured,
                Annual_Premium,
                Policy_Sales_Channel,
                Vintage,
                Vehicle_Age_lt_1_Year,
                Vehicle_Age_gt_2_Years,
                Vehicle_Damage_Yes
                ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_usvisa_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes]
            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> str:
        """
        This method takes a dataframe, loads the preprocessor and model from local files,
        transforms the data, and returns the model's prediction.
        """
        try:
            logging.info("Starting prediction...")
            
            # Get paths to the local preprocessor and model files from the config
            preprocessor_path = self.prediction_pipeline_config.preprocessor_path
            model_path = self.prediction_pipeline_config.model_file_path

            # Load the preprocessor and model objects from local files
            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info(f"Loading model from: {model_path}")
            model = load_object(file_path=model_path)

            # Use the loaded preprocessor to transform the input dataframe
            transformed_dataframe = preprocessor.transform(dataframe)
            
            # Predict using the loaded model on the transformed data
            prediction = model.predict(transformed_dataframe)
            
            return prediction

        except Exception as e:
            raise MyException(e, sys) from e