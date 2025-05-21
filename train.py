import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(file_path, target_column='Machine failure'):
    # Load preprocessed data
    data = pd.read_csv(file_path)
    print(f"Loaded data shape: {data.shape}")

    # Show how many machines have failure and how many do not
    failure_counts = data[target_column].value_counts()
    print(f"\nMachine failure counts:\n{failure_counts}")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize the model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"\nAccuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    file_path = 'data/ai_2020.csv'  # Path to preprocessed data CSV
    trained_model = train_model(file_path)
