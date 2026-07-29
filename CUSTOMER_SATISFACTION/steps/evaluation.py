import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
from zenml import step

from src.evaluation import MSE,RMSE,R2
import mlflow
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation_model(model : RegressorMixin,
                     X_test: pd.DataFrame,
                     y_test:pd.DataFrame
                     ) ->Tuple[
                         Annotated[float,'r2_score'],
                         Annotated[float,'rmse'],
                     ]:
    """
    Evaluates the model on the ingested data.
    Args:
        df:the ingested data.
    """
    try:
        prediction=model.predict(X_test)
        mse_class=MSE()
        mse=mse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("mse",mse)
        r2_class=R2()
        r2_score=r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("r2",r2_score)
        rmse_class=RMSE()
        rmse=rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse",rmse)
        return rmse, r2_score
    except Exception as e :
        logging.error(f"Error in evaluation model: {format(e)}")
        raise e