import os
from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.data_profiling import profile_data
from src.train import load_best_params, train_best_model
from src.hpt import perform_hyperparameter_optimization
from src.download_best_model import download_model
from src.datadrift import main
from src.read_json import print_file_contents
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body):
    return None

def run_pipeline():
    try:
        # Step 1: Data Ingestion
        print("Running data ingestion...")
        ingest_data()

        # Step 3: Data Drift
        print("Running data drift...")
        main()

        # Step 3: Data Preprocessing
        print("Running data preprocessing...")
        preprocess_data()
    
        # Step 4: Data Profiling
        print("Running data profiling...")
        profile_data()

        # Step 5: Hyperparameter Optimization
        print("Running hyperparameter optimization...")
        perform_hyperparameter_optimization()

        # Step 6: Final Model Training with the Best Hyperparameters
        print("Training the best model...")
        best_model_info = load_best_params()
        best_model_name = best_model_info['best_model_name']
        best_params = best_model_info['best_params']
        train_best_model(best_model_name, best_params)

        # Step 7: Download registered model
        print("Loading registered model...")
        destination_folder = 'registerd_model'
        download_model(destination_folder)

        # Step 8: Read JSON
        print("Reading JSON...")
        print_file_contents('best_model_and_params.json')

        # Send success email
        send_email(
            "Pipeline Execution Successful",
            "The pipeline executed successfully without errors."
        )
        print("Pipeline executed successfully.")

    except Exception as e:
        # Send failure email
        send_email(
            "Pipeline Execution Failed",
            f"The pipeline failed with error: {str(e)}"
        )
        print("Pipeline failed. Check your email for details.")
        raise  # Re-raise the exception to stop further execution

if __name__ == "__main__":
    print("runnig_pipeline")
    run_pipeline()