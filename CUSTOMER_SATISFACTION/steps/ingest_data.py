import logging
import pandas as pd
from zenml import step
# step1
class IngestData:
    """
    Ingesting the data from data_path  
    """
    def __init__(self, data_path : str):
        """
        Args:
            data_path:path to data
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        Ingestion the data from the data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        # Specify the date format
        # date_format = "%Y-%m-%d"  # Adjust this format to match your date format

        # Read the CSV file with the specified date format
        # df = pd.read_csv(self.data_path, index_col=0, parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format=date_format))
        return  pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path:str)->pd.DataFrame:
    """
    Ingestion the data from the data_apath

    Args:
        data_path:path to the data
    Returns:
        pd.DataFrame:the ingested data
    """
    try:
        ingest_data=IngestData(data_path)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingestion data: {e}")
        raise e