import os
import uvicorn
from dl import deeplearn
from data import data

class Config:
    def __init__(self):
        self.DATASET_FEATURESPT_PATH = os.getenv("DATASET_FEATURESPT_PATH", "./data/dataset/features.pt")
        self.DATASET_TARGETPT_PATH = os.getenv("DATASET_TARGETPT_PATH", "./data/dataset/target.pt")

        self.MODEL_TRAIN_EPOCH = int(os.getenv("MODEL_TRAIN_EPOCH", 1500))
        self.MODEL_SCALER_PATH = os.getenv("MODEL_SCALER_PATH", "./data/model/scaler.pkl")
        self.MODEL_PATH = os.getenv("MODEL_PATH", "./data/model/model.pth")

        self.FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
        self.FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8080))

if __name__ == "__main__":
    config = Config()
    if not os.path.exists(config.MODEL_PATH):
        storage = data.Storage(config.DATASET_FEATURESPT_PATH, config.DATASET_TARGETPT_PATH)
        X_train, X_test, y_train, y_test = storage.load_data()

        model = deeplearn.Model()
        model.train(config.MODEL_TRAIN_EPOCH, X_train, y_train)
        print(f"[+] Benchmark success {model.benchmark(X_test, y_test)}%")

        storage.dump_scaler(config.MODEL_SCALER_PATH)
        model.dump_model(config.MODEL_PATH)
        print(f"[+] Dump model({config.MODEL_PATH}) and dump scaler({config.MODEL_SCALER_PATH})")
    else:
        print(f"[+] Found model - \"{config.MODEL_PATH}\"")

    uvicorn.run("api.fastapi:app", host=config.FASTAPI_HOST, port=config.FASTAPI_PORT)
