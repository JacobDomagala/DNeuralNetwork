from layer import Dense, Input
import numpy as np

class Model:
    def __init__(self, layers):
        num_out = layers[0].num_features
        self.layers = layers[1:]
        for layer in self.layers:
            num_out = layer.setup(num_out)

    def __str__(self):
        str = "Model with layers:\n"
        for layer in self.layers:
            str += f"{layer}\n"
        
        return str
    
    def predict(self, X):
        A_tmp = X
        for layer in self.layers:
            A_tmp = layer.compute(A_tmp)

        return A_tmp


m = Model([Input(5), Dense(10, "relu"), Dense(20, "relu"), Dense(1, "linear")])

print(f"{m.predict(np.random.random(5))}")
