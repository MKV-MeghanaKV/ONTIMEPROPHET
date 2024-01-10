import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Data_ModelTraining','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')  
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_ordinal_features=['Weather_conditions','Road_traffic_density','City']
            categorical_ohe_features=['Type_of_order','Type_of_vehicle','Festival']
            numerical_features=['Delivery_person_Age','Delivery_person_Ratings','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Vehicle_condition','multiple_deliveries','Day_Ordered','Month_Ordered','Year_Ordered','Orderd_hour','Orderd_minute','Orderd_picked_hour','Orderd_picked_minute']
            #Define custom rankings for ohe features
            Type_of_order_cat = ['Buffet', 'Drinks', 'Meal', 'Snack']
            Type_of_vehicle_cat = ['bicycle', 'electric_scooter', 'motorcycle', 'scooter']
            Festival_cat = ['No', 'Yes']  
            #Define custom rankings for ordinal features
            Weather_conditions_cat = ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Fog'] 
            Road_traffic_density_cat = ['Low', 'Medium', 'High', 'Jam'] 
            City_cat = ['Urban', 'Metropolitian', 'Semi-Urban']      
            #Pipelines
            logging.info('Pipeline Initiated')  
            
            #Numerical Pipeline
       
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), # We chose median incase of presence of outliers.
                    ("scaler_numerical",StandardScaler(with_mean=False)) # Standardization + Min Max Scaler
                ])
            
            # Ordinal Encoding Pipeline
            
            cat_pipeline_ordinal = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories= [Weather_conditions_cat,Road_traffic_density_cat,City_cat])),
                    ('scaler_ordinal', StandardScaler(with_mean=False))
                ])
            
             # One Hot Encoding Pipeline
            cat_pipeline_ohe = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(categories = [Type_of_order_cat,Type_of_vehicle_cat,Festival_cat])),
                ('scaler_ohe', StandardScaler(with_mean=False))
                ]
                )
            
            #Combining pipelines for both numerical as well as categorical features using column transformer
            preprocessor=ColumnTransformer(transformers=[('num_pipeline',num_pipeline,numerical_features),('cat_pipeline_ordinal',cat_pipeline_ordinal,categorical_ordinal_features),('cat_pipeline_ohe',cat_pipeline_ohe,categorical_ohe_features)])
            
            return preprocessor
            
            logging.info('Pipeline Process Completed')
            
        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys) 
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}') 
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column_name='Time_taken (min)' 
            drop_columns=[target_column_name]
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            # Transforming using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on training and testing datasets')
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]  
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            ) 
            
            logging.info('Preprocessor pickle file saved')
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            
            raise CustomException(e,sys)
             
             
               
                
                
