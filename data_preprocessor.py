import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class AdultDataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        self.test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        self.data_path = os.path.join(data_dir, 'adult.data')
        self.test_path = os.path.join(data_dir, 'adult.test')
        self.df_columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.data_path):
            df = pd.read_csv(self.data_url, header=None)
            df.to_csv(self.data_path, index=False, header=False)
        if not os.path.exists(self.test_path):
            df = pd.read_csv(self.test_url, header=None, skiprows=1, comment='|')
            df.to_csv(self.test_path, index=False, header=False)

    def load(self):
        df_train = pd.read_csv(self.data_path, header=None, names=self.df_columns, skipinitialspace=True)
        # Skip the first line in test set (comment/header)
        df_test = pd.read_csv(self.test_path, header=None, names=self.df_columns, skiprows=1, skipinitialspace=True)
        # Remove trailing period in income column for test set
        df_test['income'] = df_test['income'].astype(str).str.replace('.', '', regex=False).str.strip()
        return df_train, df_test

    def combine_and_split(self, df_train, df_test, test_size=0.2, random_state=430):
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        return train_test_split(df_all, test_size=test_size, random_state=random_state, stratify=df_all['income'])

    def preprocess(self, df_train, df_test, missing_strategy='dropna', encoding='label', scale=False):
        """
        missing_strategy: 'dropna' or 'fillna'
        encoding: 'label' or 'onehot'
        scale: True/False (applies StandardScaler to numerical columns)
        """
        cat_missing_cols = ['workclass', 'occupation', 'native-country']
        # Replace '?' with NaN and trim whitespace
        for df in [df_train, df_test]:
            for col in df.columns:
                df[col] = df[col].replace('?', np.nan)
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip()
        # Handle missing values
        if missing_strategy == 'dropna':
            df_train = df_train.dropna(subset=cat_missing_cols).reset_index(drop=True)
            df_test = df_test.dropna(subset=cat_missing_cols).reset_index(drop=True)
        elif missing_strategy == 'fillna':
            for col in cat_missing_cols:
                df_train[col] = df_train[col].fillna('Unknown')
                df_test[col] = df_test[col].fillna('Unknown')
        # Identify categorical columns (excluding target)
        categorical_cols = [col for col in df_train.columns if df_train[col].dtype == 'object' and col != 'income']
        # Encode target
        le_income = LabelEncoder().fit(df_train['income'])
        df_train['income'] = le_income.transform(df_train['income'])
        df_test['income'] = le_income.transform(df_test['income'])
        # Encode features
        if encoding == 'label':
            for col in categorical_cols:
                le = LabelEncoder().fit(df_train[col])
                df_train[col] = le.transform(df_train[col])
                df_test[col] = le.transform(df_test[col])
        elif encoding == 'onehot':
            df_train = pd.get_dummies(df_train, columns=categorical_cols)
            df_test = pd.get_dummies(df_test, columns=categorical_cols)
            df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
        # Feature scaling
        if scale:
            numerical_cols = [col for col in df_train.columns if df_train[col] .dtype in ['int64', 'float64'] and col != 'income']
            scaler = StandardScaler()
            df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
            df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
        return df_train, df_test

if __name__ == "__main__":
    preprocessor = AdultDataPreprocessor()
    preprocessor.download()
    df_train, df_test = preprocessor.load()
    # Example: combine and split
    df_train_new, df_test_new = preprocessor.combine_and_split(df_train, df_test)
    # Example: preprocess with fillna and onehot encoding, with scaling
    df_train_proc, df_test_proc = preprocessor.preprocess(
        df_train_new, df_test_new, missing_strategy='fillna', encoding='label', scale=True
    )
    print('Processed train shape:', df_train_proc.shape)
    print('Processed test shape:', df_test_proc.shape)
