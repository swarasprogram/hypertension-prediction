import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate(model_path: str, df: pd.DataFrame):
    model = joblib.load(model_path)
    X = df.drop(columns=["hypertension_stage"])
    y = df["hypertension_stage"]

    preds = model.predict(X)

    cm = confusion_matrix(y, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()