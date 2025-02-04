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

    def load_data(self, file_path, encoding='ISO-8859-1'):
        return pd.read_csv(file_path, encoding=encoding)
    
    def clean_data(self, df):
        df = df.drop_duplicates()
        df['key'] = df['key'].fillna("NA")
        df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})
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
    
    def feature_selection(self, df, target_column=None, top_k=None):
        if top_k != None:
            df = df[target_column]
            X = df.drop(columns=['mode'])
            y = df['mode']
            model = RandomForestClassifier(random_state=42)
            selector = RFE(model, n_features_to_select=top_k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            return df[selected_features.tolist() + ['mode']]
        else:
            return df[target_column]
        
    def smote(self, X_train, y_train): # V3
        os = SMOTE(random_state=0, k_neighbors=5)
        os_X,os_y =os.fit_resample(X_train, y_train)
        return os_X,os_y
        
    def save_processed_data(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

if __name__ == "__main__":

    preprocessor = DataPreprocessor()
    file_path = 'C:/Meta Directory/Gatech/Machine Learning/HW/HW1/hw1_repo/datasets/raw_data/spotify-2023.csv'
    raw_data = preprocessor.load_data(file_path)
    cleaned_data = preprocessor.clean_data(raw_data)

    categorical_cols = ['key']
    encoded_data, encoded_col_names = preprocessor.encode_categorical(cleaned_data, categorical_cols)

    numerical_cols = ['artist_count', 'bpm', 'danceability_%', 'valence_%', 'energy_%',
                      'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    scaled_data = preprocessor.scale_features(encoded_data, numerical_cols)

    selected_columns = ['artist_count', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                        'instrumentalness_%', 'liveness_%', 'speechiness_%', 'mode'] + encoded_col_names
    selected_df = preprocessor.feature_selection(scaled_data, selected_columns)
    # selected_df = preprocessor.feature_selection(scaled_data, selected_columns, top_k=35) # V2

    target_column = 'mode'
    X_train, X_test, y_train, y_test = preprocessor.split_data(selected_df, target_column=target_column)
    # X_train, y_train = preprocessor.smote(X_train, y_train) # V3

    preprocessor.save_processed_data(pd.concat([X_train, y_train], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\spotify_v1/train.csv')
    preprocessor.save_processed_data(pd.concat([X_test, y_test], axis=1), 
                'C:\Meta Directory\Gatech\Machine Learning\HW\HW1\hw1_repo\datasets\cleaned_data\spotify_v1/test.csv')