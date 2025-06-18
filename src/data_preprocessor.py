import os
import pandas as pd
import numpy as np
import requests
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

    def preprocess_for_dpctgan(self, df):
        """
        Preprocess the Adult dataset specifically for DP-CTGAN synthesizer.
        
        This method prepares the data for DP-CTGAN by:
        1. Handling missing values
        2. Keeping numerical columns as numerical
        3. Cleaning categorical columns
        4. Identifying discrete columns for DP-CTGAN
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            
        Returns:
            tuple: (processed_df, discrete_columns_list) where discrete_columns_list 
                   contains names of categorical columns for DP-CTGAN
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
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Drop education-num as it's redundant with education (same as PrivBayes)
        if 'education-num' in df_processed.columns:
            df_processed = df_processed.drop('education-num', axis=1)
        
        # Identify discrete (categorical) columns for DP-CTGAN
        # DP-CTGAN can handle mixed types, so we keep object columns as categorical
        # and numerical columns as numerical
        discrete_columns = []
        
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                discrete_columns.append(col)
                # Clean categorical columns - ensure consistent string format
                df_processed[col] = df_processed[col].astype(str)
            # Keep numerical columns as numerical (age, fnlwgt, capital-gain, etc.)
        
        # Optional: Reduce cardinality of high-cardinality categorical columns
        # This helps DP-CTGAN training stability
        if 'native-country' in df_processed.columns:
            country_counts = df_processed['native-country'].value_counts()
            # Keep top 5 countries, group rest as 'Other'
            top_countries = country_counts.head(5).index.tolist()
            df_processed['native-country'] = df_processed['native-country'].apply(
                lambda x: x if x in top_countries else 'Other'
            )
        
        return df_processed, discrete_columns

    def preprocess_for_synthesizer(self, df, synthesizer_type):
        """
        Unified preprocessing interface for different synthesizers.
        
        This method provides a common interface to preprocess data for different
        synthesizer types while ensuring consistency for evaluation.
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
            
        Returns:
            tuple: (processed_df, metadata_dict) where metadata_dict contains
                   synthesizer-specific information like discrete_columns
        """
        if synthesizer_type.lower() == 'privbayes':
            processed_df = self.preprocess_for_privbayes(df)
            metadata = {
                'synthesizer_type': 'privbayes',
                'all_categorical': True,
                'discrete_columns': list(processed_df.columns)  # All columns are discrete for PrivBayes
            }
            return processed_df, metadata
            
        elif synthesizer_type.lower() == 'dpctgan':
            processed_df, discrete_columns = self.preprocess_for_dpctgan(df)
            metadata = {
                'synthesizer_type': 'dpctgan',
                'all_categorical': False,
                'discrete_columns': discrete_columns
            }
            return processed_df, metadata
            
        else:
            raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}. "
                           f"Supported types: 'privbayes', 'dpctgan'")

    def get_preprocessing_info(self, synthesizer_type):
        """
        Get information about the preprocessing applied for a specific synthesizer.
        
        This method provides metadata about the preprocessing steps, which is useful
        for ensuring consistency when preprocessing test data for evaluation.
        
        Args:
            synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
            
        Returns:
            dict: Dictionary containing preprocessing information
        """
        if synthesizer_type.lower() == 'privbayes':
            return {
                'missing_strategy': 'fillna_unknown',
                'discretization': 'quantile_based',
                'cardinality_reduction': True,
                'output_format': 'all_categorical_strings',
                'dropped_columns': ['education-num']
            }
        elif synthesizer_type.lower() == 'dpctgan':
            return {
                'missing_strategy': 'fillna_unknown',
                'discretization': 'none',
                'cardinality_reduction': True,
                'output_format': 'mixed_types',
                'dropped_columns': ['education-num']
            }
        else:
            raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}")

class CovertypeDataPreprocessor:
    """
    A class for preprocessing the UCI Covertype dataset.
    
    This class provides methods to download, load, and preprocess the Covertype dataset
    for both standard machine learning tasks and synthetic data generation.
    
    Attributes:
        data_dir (str): Directory to store downloaded data files
        data_url (str): URL for the dataset
        data_path (str): Path to the data file
        df_columns (list): List of column names for the dataset
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the CovertypeDataPreprocessor.
        
        Args:
            data_dir (str): Directory to store downloaded data files. Defaults to 'data'.
        """
        self.data_dir = data_dir
        self.data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
        self.data_path = os.path.join(data_dir, 'covtype.data.gz')
        
        # Define column names
        # First 10 columns are numerical features
        numerical_features = [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ]
        
        # Next 4 columns are Wilderness Area binary features
        wilderness_areas = [f'Wilderness_Area_{i+1}' for i in range(4)]
        
        # Next 40 columns are Soil Type binary features
        soil_types = [f'Soil_Type_{i+1}' for i in range(40)]
        
        # Combine all column names
        self.df_columns = numerical_features + wilderness_areas + soil_types + ['Cover_Type']

    def download(self):
        """
        Download the Covertype dataset from UCI repository.
        
        Downloads the dataset if it doesn't exist in the data directory.
        Creates the data directory if it doesn't exist.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.data_path):
            print(f"Downloading Covertype dataset to {self.data_path}...")
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            with open(self.data_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete!")
        else:
            print(f"Dataset already exists at {self.data_path}")

    def load(self):
        """
        Load the Covertype dataset from local file.
        
        Returns:
            pd.DataFrame: The loaded dataset with proper column names
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please run download() method first."
            )
        
        print(f"Loading dataset from {self.data_path}...")
        df = pd.read_csv(
            self.data_path,
            compression='gzip',
            header=None,
            names=self.df_columns
        )
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df

    def combine_and_split(self, df, test_size=0.2, random_state=430):
        """
        Split the Covertype dataset into train and test sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            test_size (float): Proportion of data to use for testing. Defaults to 0.2
            random_state (int): Random seed for reproducibility. Defaults to 430
            
        Returns:
            tuple: (df_train, df_test) containing the split DataFrames
        """
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['Cover_Type']  # Stratify by target column
        )

    def preprocess_for_privbayes(self, df):
        """
        Preprocess the Covertype dataset specifically for PrivBayes algorithm.
        
        This method prepares the data for PrivBayes by:
        1. Discretizing numerical columns into categorical bins using quantiles
        2. Converting all columns to categorical strings
        3. Ensuring consistent format for binary columns
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with all columns as categorical strings
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Get the first 10 numerical columns (excluding target)
        numerical_cols = self.df_columns[:10]
        
        # Discretize numerical columns using quantiles
        for col in numerical_cols:
            # Use 5 bins (quintiles) for each numerical column
            df_processed[col] = pd.qcut(
                df_processed[col],
                q=5,
                precision=0,
                duplicates='drop'
            )
            # Convert to string type
            df_processed[col] = df_processed[col].astype(str)
        
        # Convert remaining columns (binary features and target) to strings
        remaining_cols = [col for col in df_processed.columns if col not in numerical_cols]
        for col in remaining_cols:
            df_processed[col] = df_processed[col].astype(str)
        
        return df_processed

    def preprocess_for_dpctgan(self, df):
        """
        Preprocess the Covertype dataset specifically for DP-CTGAN synthesizer.
        
        This method prepares the data for DP-CTGAN by:
        1. Keeping numerical columns as numerical
        2. Identifying discrete columns (binary features and target)
        3. Ensuring proper data types for each column type
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            
        Returns:
            tuple: (processed_df, discrete_columns_list) where discrete_columns_list 
                   contains names of categorical columns for DP-CTGAN
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Identify numerical columns (first 10 columns)
        numerical_cols = self.df_columns[:10]
        
        # Identify discrete columns (Wilderness_Area, Soil_Type, and Cover_Type)
        discrete_columns = []
        
        # Add Wilderness Area columns (4 columns)
        discrete_columns.extend([f'Wilderness_Area_{i+1}' for i in range(4)])
        
        # Add Soil Type columns (40 columns)
        discrete_columns.extend([f'Soil_Type_{i+1}' for i in range(40)])
        
        # Add target column
        discrete_columns.append('Cover_Type')
        
        # Ensure numerical columns are float type
        for col in numerical_cols:
            df_processed[col] = df_processed[col].astype(float)
        
        # Ensure discrete columns are int type
        for col in discrete_columns:
            df_processed[col] = df_processed[col].astype(int)
        
        return df_processed, discrete_columns

    def preprocess_for_synthesizer(self, df, synthesizer_type):
        """
        Unified preprocessing interface for different synthesizers.
        
        This method provides a common interface to preprocess data for different
        synthesizer types while ensuring consistency for evaluation.
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            synthesizer_type (str): Type of synthesizer ('privbayes' or 'dpctgan')
            
        Returns:
            tuple: (processed_df, metadata_dict) where metadata_dict contains
                   synthesizer-specific information like discrete_columns
        """
        if synthesizer_type.lower() == 'privbayes':
            processed_df = self.preprocess_for_privbayes(df)
            metadata = {
                'synthesizer_type': 'privbayes',
                'all_categorical': True,
                'discrete_columns': list(processed_df.columns)  # All columns are discrete for PrivBayes
            }
            return processed_df, metadata
            
        elif synthesizer_type.lower() == 'dpctgan':
            processed_df, discrete_columns = self.preprocess_for_dpctgan(df)
            metadata = {
                'synthesizer_type': 'dpctgan',
                'all_categorical': False,
                'discrete_columns': discrete_columns
            }
            return processed_df, metadata
        else:
            raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}")

if __name__ == "__main__":
    # Initialize preprocessor and download data
    preprocessor = AdultDataPreprocessor()
    preprocessor.download()
    df_train, df_test = preprocessor.load()
    
    # Example: combine and split
    df_train_new, df_test_new = preprocessor.combine_and_split(df_train, df_test)
    
    # Take a small sample for testing
    df_sample = df_train_new.sample(n=100, random_state=42)
    
    print("=" * 80)
    print("UNIFIED PREPROCESSING SYSTEM TEST")
    print("=" * 80)
    
    # Test unified preprocessing interface
    print("\n1. Testing PrivBayes preprocessing through unified interface:")
    df_privbayes, metadata_privbayes = preprocessor.preprocess_for_synthesizer(df_sample, 'privbayes')
    print(f'   Shape: {df_privbayes.shape}')
    print(f'   Metadata: {metadata_privbayes}')
    print(f'   Sample columns: {list(df_privbayes.columns)[:5]}...')
    
    print("\n2. Testing DP-CTGAN preprocessing through unified interface:")
    df_dpctgan, metadata_dpctgan = preprocessor.preprocess_for_synthesizer(df_sample, 'dpctgan')
    print(f'   Shape: {df_dpctgan.shape}')
    print(f'   Metadata: {metadata_dpctgan}')
    print(f'   Discrete columns: {metadata_dpctgan["discrete_columns"]}')
    print(f'   Continuous columns: {[c for c in df_dpctgan.columns if c not in metadata_dpctgan["discrete_columns"]]}')
    
    print("\n3. Data type comparison:")
    print("   PrivBayes data types:")
    for col in df_privbayes.columns[:5]:
        print(f'     {col}: {df_privbayes[col].dtype} (sample: {df_privbayes[col].iloc[0]})')
    
    print("   DP-CTGAN data types:")
    for col in df_dpctgan.columns[:5]:
        print(f'     {col}: {df_dpctgan[col].dtype} (sample: {df_dpctgan[col].iloc[0]})')
    
    print("\n4. Preprocessing info:")
    print("   PrivBayes info:", preprocessor.get_preprocessing_info('privbayes'))
    print("   DP-CTGAN info:", preprocessor.get_preprocessing_info('dpctgan'))
    
    print("\n" + "=" * 80)
    print("COVERTYPE DATASET TEST")
    print("=" * 80)
    
    # Initialize Covertype preprocessor
    covertype_preprocessor = CovertypeDataPreprocessor()
    
    # Test download
    print("\n1. Testing download:")
    covertype_preprocessor.download()
    
    # Test load
    print("\n2. Testing load:")
    df_covertype = covertype_preprocessor.load()
    print(f"   Loaded dataset shape: {df_covertype.shape}")
    print(f"   Column names: {df_covertype.columns.tolist()}")
    
    # Take a small sample for testing
    df_covertype_sample = df_covertype.sample(n=100, random_state=42)
    
    # Test PrivBayes preprocessing
    print("\n3. Testing PrivBayes preprocessing:")
    df_covertype_privbayes = covertype_preprocessor.preprocess_for_privbayes(df_covertype_sample)
    print(f"   Processed shape: {df_covertype_privbayes.shape}")
    print("\n   Data types after PrivBayes preprocessing:")
    for col in df_covertype_privbayes.columns[:5]:
        print(f"     {col}: {df_covertype_privbayes[col].dtype} (sample: {df_covertype_privbayes[col].iloc[0]})")
    
    # Test DP-CTGAN preprocessing
    print("\n4. Testing DP-CTGAN preprocessing:")
    df_covertype_dpctgan, discrete_cols = covertype_preprocessor.preprocess_for_dpctgan(df_covertype_sample)
    print(f"   Processed shape: {df_covertype_dpctgan.shape}")
    print(f"   Number of discrete columns: {len(discrete_cols)}")
    print("\n   Data types after DP-CTGAN preprocessing:")
    print("   Numerical columns (first 5):")
    for col in df_covertype_dpctgan.columns[:5]:
        print(f"     {col}: {df_covertype_dpctgan[col].dtype} (sample: {df_covertype_dpctgan[col].iloc[0]})")
    
    print("\n   Discrete columns (first 5):")
    for col in discrete_cols[:5]:
        print(f"     {col}: {df_covertype_dpctgan[col].dtype} (sample: {df_covertype_dpctgan[col].iloc[0]})")
    
    print("\n5. Column type distribution:")
    numerical_cols = [col for col in df_covertype_dpctgan.columns if col not in discrete_cols]
    print(f"   Numerical columns: {len(numerical_cols)}")
    print(f"   Discrete columns: {len(discrete_cols)}")
    
    # Verify discrete columns are integers
    print("\n6. Discrete column type verification:")
    non_int_discrete = [col for col in discrete_cols if df_covertype_dpctgan[col].dtype != 'int64']
    if non_int_discrete:
        print(f"   WARNING: Found {len(non_int_discrete)} discrete columns that are not integers:")
        for col in non_int_discrete[:5]:
            print(f"     {col}: {df_covertype_dpctgan[col].dtype}")
    else:
        print("   All discrete columns are properly converted to integers")
    
    print("\n" + "=" * 80)
    print("COVERTYPE PREPROCESSING SYSTEM READY FOR PIPELINE!")
    print("=" * 80)
