from pipelines.training_pipline import train_pipeline
from zenml.client import Client
if __name__=="__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="C:/Users/zitou/OneDrive/Documents/python/Courses/MLOpd/project/CUSTOMER_SATISFACTION/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri file:C:\Users\zitou\AppData\Roaming\zenml\local_stores\afb257f6-b6b6-4340-95a3-bf37ac3c9a78\mlruns