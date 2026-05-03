import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import SGD

class Model:
    def __init__(self):
        self.model = Sequential(Linear(4, 16), ReLU(), Linear(16, 16), ReLU(), Linear(16, 3))
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = CrossEntropyLoss()

    def train(self, epoch: int, X: any, y: any):
        losses = []
        for _ in range(epoch):
            self.model.train()
            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, y)
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses

    def dump_model(self, path: str):
        return torch.save(self.model.state_dict(), path)

    def predict(self, X: any):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(dim=1)

    def benchmark(self, X: any, y: any):
        self.model.eval()
        with torch.no_grad():
            score = ((self.model(X).argmax(1) == y).float().mean()) * 100
        return score.item()
