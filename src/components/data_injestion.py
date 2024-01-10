import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Get the current working directory
current_directory = os.getcwd()

# Specify the name of the new folder
new_folder_name = "Data_ModelTraining"

#Parent Directory path
Parent_dir=os.path.join(current_directory,new_folder_name)

## Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join(Parent_dir,'train.csv')
    test_data_path:str=os.path.join(Parent_dir,'test.csv')
    raw_data_path:str=os.path.join(Parent_dir,'raw.csv')
    
    
## create a class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Methods Starts')
        try:
            df=pd.read_csv(os.path.join(r'C:\Users\MEGHANA\Desktop\Data Science Projects Ineuorn\New folder\notebooks\data','finalTrain.csv'))
            logging.info('Dataset read as pandas Dataframe')
            
            ## Removing unnecessary columns
            logging.info("Removing unnecessary columns")
            df = df.drop(labels=["ID"],axis=1)
            df = df.drop(labels=["Delivery_person_ID"],axis=1)
            
            df["Order_Date"] = pd.to_datetime(df["Order_Date"],format='%d-%m-%Y')
            df["Day_Ordered"] = pd.to_datetime(df["Order_Date"],format='%d-%m-%Y').dt.day
            df["Month_Ordered"] = pd.to_datetime(df["Order_Date"],format='%d-%m-%Y').dt.month
            df["Year_Ordered"] = pd.to_datetime(df["Order_Date"],format='%d-%m-%Y').dt.year
            df.drop("Order_Date",axis=1,inplace=True)
            
            df['Time_Orderd'].fillna(df['Time_Orderd'].mode().iloc[0],inplace=True)
            time_pattern=r'^\d{2}:\d{2}$'
            Match_value_Time_Orderd=df['Time_Orderd'].str.match(time_pattern)
            df.loc[~Match_value_Time_Orderd,'Time_Orderd']=df['Time_Orderd'].mode()
            df['Time_Orderd']=pd.to_datetime(df['Time_Orderd'],format='%H:%M')
            df['Orderd_hour']=pd.to_datetime(df['Time_Orderd'],format='%H:%M').dt.hour
            df['Orderd_minute']=pd.to_datetime(df['Time_Orderd'],format='%H:%M').dt.minute
            df.drop('Time_Orderd',axis=1,inplace=True)
            
            
            df['Time_Order_picked'].fillna(df['Time_Order_picked'].mode().iloc[0],inplace=True)
            Match_value_Time_Orderd_picked=df['Time_Order_picked'].str.match(time_pattern)
            df.loc[~Match_value_Time_Orderd_picked,'Time_Order_picked']=df['Time_Order_picked'].mode()
            df['Time_Order_picked']=pd.to_datetime(df['Time_Order_picked'],format='%H:%M')
            df['Orderd_picked_hour']=pd.to_datetime(df['Time_Order_picked'],format='%H:%M').dt.hour
            df['Orderd_picked_minute']=pd.to_datetime(df['Time_Order_picked'],format='%H:%M').dt.minute
            df.drop('Time_Order_picked',axis=1,inplace=True)
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Started Train Test Split')
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=30)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of Data is Completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info('Exception Occured At Data Ingestion Stage')
            raise CustomException(e,sys)
    
      
            