import flwr as fl

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config): return []
    def set_parameters(self, params): pass
    def fit(self, params, config): return [], 0, {}
    def evaluate(self, params, config): return 0.0, 0, {}

print(hasattr(FLClient(), "to_client"))  
