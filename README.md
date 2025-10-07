# End-to-End MLOps Pipeline for Vehicle Insurance Claim Prediction

This project implements a complete end-to-end MLOps pipeline for a machine learning model that predicts vehicle insurance claims. The pipeline covers everything from data ingestion and validation to model training, evaluation, and deployment via a web application.

## Features

-   **Automated Training Pipeline**: Orchestrates all steps from data extraction to model registration.
-   **Data Ingestion**: Fetches data from a MongoDB database.
-   **Data Validation**: Ensures data quality and schema integrity.
-   **Data Transformation**: Preprocesses data, handles categorical features, and prepares it for modeling.
-   **Model Training**: Trains a classification model to predict claim approvals.
-   **Model Evaluation**: Compares the newly trained model with the production model to ensure performance improvement.
-   **Model Pusher**: Pushes the best model to an AWS S3 bucket for production use.
-   **Prediction Pipeline**: Serves the trained model for real-time predictions.
-   **Web Application**: A user-friendly interface built with Streamlit to interact with the model.
-   **Dockerized**: The application is containerized using Docker for easy deployment.
-   **CI/CD**: Includes setup for continuous integration and deployment (structure available in `.github/workflows`).

## Project Structure

```
├── artifacts/              # Stores output artifacts from the pipeline (models, datasets, etc.)
├── config/                 # Configuration files (schema.yaml, model.yaml)
├── logs/                   # Log files for monitoring and debugging
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code for the MLOps pipeline
│   ├── components/         # Individual pipeline components (data ingestion, training, etc.)
│   ├── pipline/            # Pipeline orchestration (training and prediction)
│   ├── entity/             # Configuration and artifact entity classes
│   ├── data_access/        # Scripts for accessing data (e.g., from MongoDB)
│   ├── cloud_storage/      # Modules for interacting with AWS S3
│   └── utils/              # Utility functions
├── app.py                  # Streamlit application for prediction
├── demo.py                 # Script to run the training pipeline
├── Dockerfile              # Docker configuration for the application
├── requirements.txt        # Python dependencies
└── setup.py                # Setup script for installing the project as a package
```

## Technology Stack

-   **Programming Language**: Python 3.10
-   **ML/Data Science**: Scikit-learn, Pandas, NumPy
-   **Data Storage**: MongoDB
-   **Model Registry**: AWS S3
-   **Web Framework**: Streamlit
-   **Containerization**: Docker

## Setup and Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd MLOPS-End-to-End-Pipeline-Implementation
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    conda create -n proj1 python=3.10 -y
    conda activate proj1
    ```

3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    You will need to configure credentials for MongoDB and AWS. Create a `.env` file in the root directory and add the following:
    ```
    MONGODB_URL="<your_mongodb_connection_string>"
    AWS_ACCESS_KEY_ID="<your_aws_access_key>"
    AWS_SECRET_ACCESS_KEY="<your_aws_secret_key>"
    ```

## How to Run

### 1. Run the Training Pipeline

To execute the entire training pipeline, run the `demo.py` script:

```sh
python demo.py
```

This will trigger the data ingestion, validation, transformation, model training, evaluation, and model pusher stages. The resulting artifacts will be stored in the `artifacts/` directory.

### 2. Run the Prediction Web App

To start the Streamlit application for making predictions:

```sh
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to use the application. You can input vehicle and customer data to get a claim prediction. The app also provides an option to trigger the training pipeline directly from the UI.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
