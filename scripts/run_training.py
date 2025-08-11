import sys
import os

# Añade el directorio raíz del proyecto al sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.train_model import train_pipeline

import mlflow
from src.train_model import train_pipeline

mlflow.set_experiment("FakeNewsDetection")

with mlflow.start_run():
    # Entrena y registra todo desde train_pipeline
    train_pipeline()
