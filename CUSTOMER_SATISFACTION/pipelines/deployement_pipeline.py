import numpy as np
import pandas as pd
# from materializer.custom_materializer import cs_materializer

from zenml import pipeline ,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT 
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluation_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model


docker_settings= DockerSettings(required_integrations=[MLFLOW])

def inference_pipeline():
    pass

class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config"""
    min_accuracy : float = 0

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implements a simple model deployment trigger that look at the input model accuacy and decides if it is good to deploy or not"""
    return accuracy>= config.min_accuracy
class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name:str
    running: bool =True
    model_name : str ="model"

@pipeline(enable_cache=False,settings={"docker": docker_settings})
def continuous_deployement_pipeline(
    data_path : str, 
    min_accuracy: float = 0.92,
    workers : int=1,
    timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    import mlflow
    mlflow.set_experiment("continuous_deployement_pipeline")

    df= ingest_df(data_path=data_path)
    X_train,X_test,y_train,y_test= clean_df(df)
    model=train_model(X_train,X_test,y_train,y_test)
    r2_score, rmse=evaluation_model(model,X_test,y_test)
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
