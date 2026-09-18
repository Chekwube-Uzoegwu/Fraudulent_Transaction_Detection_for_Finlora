import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient
from src.utility.mlflow_setup import setup_mlflow


def load_registered_model(model_name="Fraud_Detection_LightGBM_Pipeline", model_version="latest"):
    """
    Load a registered model from the MLflow Model Registry.
    """
    setup_mlflow()
    client = MlflowClient()

    if model_version == "latest":
        # Find the highest version number registered under this name
        versions = client.search_model_versions(f"name='{model_name}'")
        model_version = max(versions, key=lambda v: int(v.version)).version

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model