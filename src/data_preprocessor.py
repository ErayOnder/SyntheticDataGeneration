import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class AdultDataPreprocessor:
    """
    A class for preprocessing the UCI Adult dataset.
    
    This class provides methods to download, load, and preprocess the Adult dataset
    for both standard machine learning tasks and the PrivBayes algorithm.
    
    Attributes:
        data_dir (str): Directory to store downloaded data files
        data_url (str): URL for the training data
        test_url (str): URL for the test data
        data_path (str): Path to the training data file
        test_path (str): Path to the test data file
        df_columns (list): List of column names for the dataset
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the AdultDataPreprocessor.
        
        Args:
            data_dir (str): Directory to store downloaded data files. Defaults to 'data'.
        """
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
        """
        Download the Adult dataset from UCI repository.
        
        Downloads both training and test datasets if they don't exist in the data directory.
        Creates the data directory if it doesn't exist.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.data_path):
            df = pd.read_csv(self.data_url, header=None)
            df.to_csv(self.data_path, index=False, header=False)
        if not os.path.exists(self.test_path):
            df = pd.read_csv(self.test_url, header=None, skiprows=1, comment='|')
            df.to_csv(self.test_path, index=False, header=False)

    def load(self):
        """
        Load the Adult dataset from local files.
        
        Returns:
            tuple: (df_train, df_test) containing the training and test DataFrames
        """
        df_train = pd.read_csv(self.data_path, header=None, names=self.df_columns, skipinitialspace=True)
        # Skip the first line in test set (comment/header)
        df_test = pd.read_csv(self.test_path, header=None, names=self.df_columns, skiprows=1, skipinitialspace=True)
        # Remove trailing period in income column for test set
        df_test['income'] = df_test['income'].astype(str).str.replace('.', '', regex=False).str.strip()
        return df_train, df_test

    def combine_and_split(self, df_train, df_test, test_size=0.2, random_state=430):
        """
        Combine training and test sets, then split into new train/test sets.
        
        Args:
            df_train (pd.DataFrame): Original training data
            df_test (pd.DataFrame): Original test data
            test_size (float): Proportion of data to use for testing. Defaults to 0.2
            random_state (int): Random seed for reproducibility. Defaults to 430
            
        Returns:
            tuple: (df_train_new, df_test_new) containing the new split DataFrames
        """
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        return train_test_split(df_all, test_size=test_size, random_state=random_state, stratify=df_all['income'])

    def standard_preprocess(self, df_train, df_test, missing_strategy='dropna', encoding='label', scale=False):
        """
        Standard preprocessing for machine learning models.
        
        This method handles missing values, encodes categorical variables, and optionally
        scales numerical features. It works on copies of the input DataFrames to prevent
        modifying the original data.
        
        Args:
            df_train (pd.DataFrame): Training data
            df_test (pd.DataFrame): Test data
            missing_strategy (str): How to handle missing values. Options: 'dropna' or 'fillna'
            encoding (str): How to encode categorical variables. Options: 'label' or 'onehot'
            scale (bool): Whether to scale numerical features. Defaults to False
            
        Returns:
            tuple: (processed_train_df, processed_test_df) containing the preprocessed DataFrames
        """
        # Create copies to avoid modifying original DataFrames
        df_train = df_train.copy()
        df_test = df_test.copy()
        
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
            numerical_cols = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64'] and col != 'income']
            scaler = StandardScaler()
            df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
            df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
        return df_train, df_test

    def preprocess_for_privbayes(self, df):
        """
        Preprocess the Adult dataset specifically for PrivBayes algorithm.
        
        This method prepares the data for PrivBayes by:
        1. Handling missing values
        2. Discretizing numerical columns into categorical bins
        3. Reducing cardinality of high-cardinality categorical columns
        4. Converting all columns to categorical strings
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with all columns as categorical strings
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Replace '?' with NaN and trim whitespace
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].replace('?', np.nan)
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].str.strip()
        
        # Handle missing values for specific columns
        cat_missing_cols = ['workclass', 'occupation', 'native-country']
        for col in cat_missing_cols:
            df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Drop education-num as it's redundant with education
        df_processed = df_processed.drop('education-num', axis=1)
        
        # Discretize numerical columns
        
        # Age: 5 bins (quintiles)
        df_processed['age'] = pd.qcut(df_processed['age'], q=5, precision=0, duplicates='drop')
        df_processed['age'] = df_processed['age'].astype(str)
        
        # fnlwgt: 10 bins (deciles)
        df_processed['fnlwgt'] = pd.qcut(df_processed['fnlwgt'], q=10, precision=0, duplicates='drop')
        df_processed['fnlwgt'] = df_processed['fnlwgt'].astype(str)
        
        # capital-gain: Handle skewed distribution with many zeros
        capital_gain_90th = df_processed['capital-gain'].quantile(0.9)
        df_processed['capital-gain'] = df_processed['capital-gain'].apply(
            lambda x: 'None' if x == 0 else ('Low' if x <= capital_gain_90th else 'High')
        )
        
        # capital-loss: Handle skewed distribution with many zeros
        capital_loss_90th = df_processed['capital-loss'].quantile(0.9)
        df_processed['capital-loss'] = df_processed['capital-loss'].apply(
            lambda x: 'None' if x == 0 else ('Low' if x <= capital_loss_90th else 'High')
        )
        
        # hours-per-week: 5 bins (quintiles)
        df_processed['hours-per-week'] = pd.qcut(df_processed['hours-per-week'], q=5, precision=0, duplicates='drop')
        df_processed['hours-per-week'] = df_processed['hours-per-week'].astype(str)
        
        # Reduce cardinality of native-country
        # Find top 2 most frequent countries
        country_counts = df_processed['native-country'].value_counts()
        top_2_countries = country_counts.head(2).index.tolist()
        
        # Group all other countries into 'Other'
        df_processed['native-country'] = df_processed['native-country'].apply(
            lambda x: x if x in top_2_countries else 'Other'
        )
        
        # Ensure all columns are categorical strings
        for col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
        
        return df_processed

if __name__ == "__main__":
    # Initialize preprocessor and download data
    preprocessor = AdultDataPreprocessor()
    preprocessor.download()
    df_train, df_test = preprocessor.load()
    
    # Example: combine and split
    df_train_new, df_test_new = preprocessor.combine_and_split(df_train, df_test)
    
    # Example: standard preprocessing
    df_train_proc, df_test_proc = preprocessor.standard_preprocess(
        df_train_new, df_test_new, missing_strategy='fillna', encoding='label', scale=True
    )
    print('Processed train shape:', df_train_proc.shape)
    print('Processed test shape:', df_test_proc.shape)
    
    # Example: PrivBayes preprocessing
    print("\n--- PrivBayes Preprocessing ---")
    df_privbayes = preprocessor.preprocess_for_privbayes(df_train_new)
    print('PrivBayes processed shape:', df_privbayes.shape)
    
    print('\nFirst 5 rows with column names:')
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Don't wrap wide tables
    pd.set_option('display.max_colwidth', None) # Don't truncate cell contents
    print(df_privbayes.head().to_string())
    
    print('\nDataFrame info:')
    print(df_privbayes.info())
    
    print('\nUnique values per column:')
    for col in df_privbayes.columns:
        unique_vals = df_privbayes[col].unique()
        print(f'\n{col}: {len(unique_vals)} unique values')
        print('Values:', sorted(unique_vals))
