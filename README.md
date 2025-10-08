# MLOps Project - Vehicle Insurance Claim Prediction

Welcome to this MLOps project, which demonstrates a robust, end-to-end pipeline for a vehicle insurance claim prediction task. This project showcases data ingestion from MongoDB, automated model training, and a simple web interface for predictions. The goal is to provide a clear example of building and managing a machine learning pipeline for real-world applications.

---

## 📁 Project Setup and Structure

### Step 1: Project Template
- Start by executing the `template.py` file to create the initial project template, which includes the required folder structure and placeholder files.

### Step 2: Package Management
- Write the setup for importing local packages in `setup.py`.

### Step 3: Virtual Environment and Dependencies
- Create a virtual environment and install required dependencies from `requirements.txt`:
  ```bash
  conda create -n vehicle python=3.10 -y
  conda activate vehicle
  pip install -r requirements.txt
  ```
- Verify the local packages by running:
  ```bash
  pip list
  ```

### Project Structure
```
.
├── .github/workflows/         # GitHub Actions CI/CD (if applicable)
├── notebooks/                 # Jupyter notebooks for EDA and experiments
│   └── mongoDB_demo.ipynb
├── src/                       # Source code for the project
│   ├── __init__.py
│   ├── components/            # Core pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── config/                # Configuration files
│   │   └── schema.yaml
│   ├── constants/             # Project constants
│   ├── entity/                # Entity definitions (configs, artifacts)
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/             # Custom exception handling
│   ├── logger/                # Logging setup
│   ├── pipeline/              # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   └── utils/                 # Utility functions
│       └── main_utils.py
├── app.py                     # Main application file (Streamlit)
├── demo.py                    # Script to run the training pipeline
├── requirements.txt           # Project dependencies
├── setup.py                   # Setup script for the project package
└── README.md                  # Project documentation
```

---

## 📊 MongoDB Setup and Data Management

### Step 4: MongoDB Atlas Configuration
1. Sign up for [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a new project.
2. Set up a free M0 cluster, configure the username and password, and allow access from any IP address (`0.0.0.0/0`).
3. Retrieve the MongoDB connection string for Python and save it (replace `<password>` with your password).

### Step 5: Pushing Data to MongoDB
1. Create a folder named `notebook`, add the dataset, and create a notebook file `mongoDB_demo.ipynb`.
2. Use the notebook to push data to the MongoDB database.
3. Verify the data in MongoDB Atlas under Database > Browse Collections.

---

## 📝 Logging, Exception Handling, and EDA

### Step 6: Set Up Logging and Exception Handling
- Create logging and exception handling modules. Test them on a demo file `demo.py`.

### Step 7: Exploratory Data Analysis (EDA)
- Analyze and engineer features in a notebook for further processing in the pipeline.

---

## ⚙️ ML Pipeline Execution

The core of this project is an automated pipeline that handles everything from data ingestion to model training.

### Step 8: Configure Environment Variables
- The pipeline requires the MongoDB connection string. Set it as an environment variable:
  ```bash
  # For Bash/Zsh
  export MONGODB_URL="mongodb+srv://<username>:<password>...."
  
  # For Windows Powershell
  $env:MONGODB_URL = "mongodb+srv://<username>:<password>...."
  ```

### Step 9: Run the Training Pipeline
- Execute the main script to run the entire pipeline:
  ```bash
  python demo.py
  ```
- This will trigger the following components sequentially:
    1.  **Data Ingestion**: Fetches data from your MongoDB collection.
    2.  **Data Validation**: Validates data against a predefined schema.
    3.  **Data Transformation**: Preprocesses the data and prepares it for training.
    4.  **Model Training**: Trains a machine learning model and saves the artifact.

---

## 🚀 Prediction with Streamlit UI

A simple web interface is available to interact with the trained model.

### Step 10: Run the Application
- Once the training pipeline has successfully generated a model, start the Streamlit application:
  ```bash
  streamlit run app.py
  ```
- Open your browser and navigate to the local URL provided to make predictions.

---

## 🎯 Project Workflow Summary

1.  **Data Ingestion** (from MongoDB) ➔ **Data Validation** ➔ **Data Transformation**
2.  **Model Training** ➔ **Model Artifact Saved**
3.  **Prediction** (via Streamlit UI)

---

## 💬 Connect
If you found this project helpful or have any questions, feel free to reach out!

Contact: kaloorishivaprasad@gmail.com
