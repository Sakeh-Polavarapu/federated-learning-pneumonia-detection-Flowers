import flwr as fl
import torch
from utils import get_model, load_data
from train_utils import evaluate as evaluate_model

def get_evaluate_fn():
    model = get_model()
    _, test_loader = load_data(client_id=0, num_clients=1)

    def evaluate(server_round, parameters, config):
        state_dict = model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        model.load_state_dict(state_dict)
        accuracy = evaluate_model(model, test_loader)
        print(f"🌐 [Server] Round {server_round} Central Eval Accuracy: {accuracy:.2f}%")
        return 0.0, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            evaluate_fn=get_evaluate_fn()
        )
    )
