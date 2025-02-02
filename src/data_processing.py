import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)

    def load_data(self, file_path, sep='\t'):
        return pd.read_csv(file_path, sep=sep)

    def clean_data(self, df):
        df = df.drop_duplicates()

        for column in df.columns:
            if df[column].dtype == np.number:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
        return df

    def encode_categorical(self, df, categorical_columns):
        encoded_cols = self.encoder.fit_transform(df[categorical_columns])
        encoded_col_names = self.encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names)
        
        df = df.drop(columns=categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df, list(encoded_col_names)

    def scale_features(self, df, numerical_columns):
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def feature_selection(self, df, target_column=None):
        if target_column != None:
            return df[target_column]

    def save_processed_data(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

if __name__ == "__main__":

    preprocessor = DataPreprocessor()
    file_path = 'C:/Meta Directory/Gatech/Machine Learning/HW/HW1/hw1_repo/datasets/raw_data/marketing_campaign.csv'
    raw_data = preprocessor.load_data(file_path)
    cleaned_data = preprocessor.clean_data(raw_data)

    categorical_cols = ['Education', 'Marital_Status']
    encoded_data, encoded_col_names = preprocessor.encode_categorical(cleaned_data, categorical_cols)

    numerical_cols = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                      'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                      'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
                      'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
                      'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                      'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue']
    scaled_data = preprocessor.scale_features(encoded_data, numerical_cols)

    selected_columns = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                      'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                      'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
                      'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
                      'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                      'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'] + encoded_col_names
    selected_df = preprocessor.feature_selection(scaled_data, selected_columns)

    target_column = 'Response'
    X_train, X_test, y_train, y_test = preprocessor.split_data(selected_df, target_column=target_column)

    preprocessor.save_processed_data(pd.concat([X_train, y_train], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\mkt_camp/train.csv')
    preprocessor.save_processed_data(pd.concat([X_test, y_test], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\mkt_camp/test.csv')