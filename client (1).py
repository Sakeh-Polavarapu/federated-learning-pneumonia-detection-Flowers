import flwr as fl
import torch
import time
from utils import get_model, load_data
from train_utils import train, evaluate  

time.sleep(5)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = get_model()
        self.trainloader, self.testloader = load_data(client_id=int(cid), num_clients=3)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print(f"Client {self.cid} fit() triggered")
        self.set_parameters(parameters)
        print(f"Client {self.cid} parameters set")
        train(self.model, self.trainloader, epochs=5, lr=0.05)
        print(f"Client {self.cid} training completed")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"Client {self.cid} evaluate() triggered")
        self.set_parameters(parameters)
        accuracy = evaluate(self.model, self.testloader)
        return float(0.0), len(self.testloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    fl.client.start_numpy_client(server_address="server:8080", client=FlowerClient(client_id))
