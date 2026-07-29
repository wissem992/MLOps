from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model config"""
    model_name: str="LinearRegression"