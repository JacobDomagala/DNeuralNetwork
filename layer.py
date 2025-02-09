import numpy as np
import functions

class Base:
    def __init__(self, is_debug):
        self.is_debug = is_debug

class Input(Base):
    def __init__(self, num_features):
        self.num_features = num_features

    def __str__(self):
        return f"Input layer with {self.num_features} features"
    
    
class Dense(Base):
    def __init__(self, num_neurons, activation_func):
        super().__init__(True)
        self.num_neurons = num_neurons
        self.activation_func = activation_func
    
    def setup(self, num_features):
        self.W = np.random.random((num_features, self.num_neurons))
        self.b = np.random.random(self.num_neurons)
        
        return self.num_neurons
    
    def compute(self, A_in):
        A = A_in @ self.W + self.b

        if self.activation_func == "relu":
            A = functions.relu(A)
        elif self.activation_func == "linear":
            pass
        elif self.activation_func == "sigmoid":
            A = functions.sigmoid(A)
        else:
            print(f"ERROR: Wrong activation fiunction {self.activation_func}")

        if(self.is_debug):
            print(f"DEBUG: Computing A @ W for shapes={A_in.shape} and {self.W.shape}. The output shape is {A.shape}")
        
        return A
    
    def __str__(self):
        return f"Dense layer with {self.num_neurons} neurons with shape={self.W.shape}"

