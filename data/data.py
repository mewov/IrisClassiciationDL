import torch
import joblib

class Storage:
    def __init__(self, features_pt_path: str, target_pt_path: str):
        self.features = features_pt_path
        self.target = target_pt_path
        self.mean = 0
        self.std = 0

    def load_data(self):
        X = torch.load(self.features, weights_only=False)
        y = torch.load(self.target, weights_only=False).long()

        X_train, X_test = X[:100], X[100:]
        y_train, y_test = y[:100], y[100:]

        X_mean = X_train.mean(dim=0, keepdim=True)
        X_std = X_train.std(dim=0, keepdim=True)
        X_std[X_std == 0] = 1.0

        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        self.std = X_std
        self.mean = X_mean
        return X_train, X_test, y_train, y_test
    
    def dump_scaler(self, path: str):
        joblib.dump({
            "mean": self.mean,
            "std": self.std
        }, path)