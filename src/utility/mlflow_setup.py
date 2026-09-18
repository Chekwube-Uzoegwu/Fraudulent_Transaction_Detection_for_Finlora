import dagshub
import mlflow


def setup_mlflow():
    """
    Initializes MLflow tracking for the project.
    """
    dagshub.init(repo_owner='uzoegwuchekwubechris',
                 repo_name='Fraudulent_Transaction_Detection_for_Finlora',
                 mlflow=True)

    # Set experiment
    mlflow.set_experiment("Fraudulent_Transaction_Detection_for_Finlora_Company")