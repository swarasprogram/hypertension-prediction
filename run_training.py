import pandas as pd
import os
from src.preprocess import clean_and_engineer 
from src.train import train_model

def run():
    print("Loading data...")
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/raw/cardio_train.csv', sep=';')
    print("Cleaning and engineering data...")
    df_clean = clean_and_engineer(df)
    print("Training model...")
    train_model(df_clean, 'models/hypertension_model.joblib')
    print("Model retrained successfully.")

if __name__ == "__main__":
    run()
