import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
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

    def feature_selection(self, df, target_column=None, top_k=None): # V4
        if top_k != None:
            df = df[target_column]
            X = df.drop(columns=['Response'])
            y = df['Response']
            model = RandomForestClassifier(random_state=42)
            selector = RFE(model, n_features_to_select=top_k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            return df[selected_features.tolist() + ['Response']]
        else:
            return df[target_column]

    def save_processed_data(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def feature_engineering(self, data): # V2
        data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)
        dates = []
        for i in data["Dt_Customer"]:
            i = i.date()
            dates.append(i)  
        days = []
        d1 = max(dates)
        for i in dates:
            delta = d1 - i
            days.append(delta)
        data["Customer_For"] = days
        data["Customer_For"] = data["Customer_For"].dt.days.astype(int)
        data["Age"] = 2021-data["Year_Birth"]
        data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]
        data["Children"]=data["Kidhome"]+data["Teenhome"]
        data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone"})
        data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]
        data["Is_Parent"] = np.where(data.Children> 0, 1, 0)
        data["Age"] = data["Age"].where(data["Age"] <= 90, 90)
        data["Income"] = data["Income"].where(data["Income"] <= 600000, 600000)
        return data
    
    def smote(self, X_train, y_train): # V3
        os = SMOTE(random_state=0, k_neighbors=5)
        os_X,os_y =os.fit_resample(X_train, y_train)
        return os_X,os_y

if __name__ == "__main__":

    preprocessor = DataPreprocessor()
    file_path = 'C:/Meta Directory/Gatech/Machine Learning/HW/HW1/hw1_repo/datasets/raw_data/marketing_campaign.csv'
    raw_data = preprocessor.load_data(file_path)
    cleaned_data = preprocessor.clean_data(raw_data)
    cleaned_data = preprocessor.feature_engineering(cleaned_data) # V2

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
                      'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response', 
                      'Customer_For', 'Age', 'Spent', 'Children', 'Family_Size', 'Is_Parent'] + encoded_col_names
    # selected_df = preprocessor.feature_selection(scaled_data, selected_columns)
    selected_df = preprocessor.feature_selection(scaled_data, selected_columns, top_k=35) # V4

    target_column = 'Response'
    X_train, X_test, y_train, y_test = preprocessor.split_data(selected_df, target_column=target_column)
    X_train, y_train = preprocessor.smote(X_train, y_train) # V3

    preprocessor.save_processed_data(pd.concat([X_train, y_train], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\mkt_camp_v4/train.csv')
    preprocessor.save_processed_data(pd.concat([X_test, y_test], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\mkt_camp_v4/test.csv')