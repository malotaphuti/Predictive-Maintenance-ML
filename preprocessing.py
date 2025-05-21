import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # 1. Load data
    data = pd.read_csv(file_path)
    print(f"Initial data shape: {data.shape}")

    # 2. Explore data briefly
    print(data.info())
    print(data.isnull().sum())  # missing values per column

    # 3. Separate target column to avoid transforming it
    target_column = 'Machine failure'
    target = data[target_column]

    # 4. Handle missing values
    # Exclude target column from numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop(target_column)
    imputer_num = SimpleImputer(strategy='median')
    data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])

    # For categorical features, fill missing with mode (most frequent)
    categorical_cols = data.select_dtypes(include=['object']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

    # 5. Encode categorical variables (Label Encoding for simplicity)
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 6. Feature scaling (StandardScaler), exclude target
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # 7. Reattach target column without modification
    data[target_column] = target

    print("Preprocessing completed.")
    return data

if __name__ == "__main__":
    # The input file path
    file_name = 'data/ai_2020.csv' 
    
    processed_data = preprocess_data(file_name)
    
    # Save the processed data back to the original file name
    processed_data.to_csv(file_name, index=False)
    print(f"Processed data saved to '{file_name}' (original file overwritten).")
