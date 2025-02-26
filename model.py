import numpy as np

class Layer:
    def __init__(self, units = 0):
        self.alpha = 0.01
        self.units = units
    def forward(self, A_in):
        return A_in
    def backward(self, grad):
        return grad
    def compile(self, num_input):
        return self.units

class Input(Layer):
    def __init__(self, units):
        super().__init__(units)

    def __repr__(self):
        return f"Input(neurons={self.units})"

class Dense(Layer):
    def __init__(self, units):
        super().__init__(units)

    def compile(self, num_input):
        self.W = np.random.random((num_input, self.units)) * np.sqrt(2 / num_input)
        self.b = np.zeros(self.units)

        return self.units

    def __repr__(self):
        return f"Dense(neurons={self.units})"

    def forward(self, A_in):
        self.data = A_in
        return A_in @ self.W + self.b

    def backward(self, grad):
        self.grad = self.data.T @ grad
        return_grad = grad @ self.W.T
        self.W -= self.alpha * self.grad
        self.b -= self.alpha * grad.sum(axis=0)
        return return_grad


class ReLu(Layer):
    def __repr__(self):
        return f"ReLu"
    def compile(self, num_units):
        return num_units
    def forward(self, A_in):
        self.data = A_in
        return np.maximum(0, A_in)
    def backward(self, grad):
        mask = self.data > 0
        self.grad = mask * grad
        return self.grad

class Model:
    def __init__(self, layers_):
        self.layers = layers_
        assert(isinstance(self.layers, list))
        assert(len(self.layers) > 1)
        for l in layers_:
            assert(isinstance(l, Layer))
        self.compiled = False


    def compile(self):
        num_units = 0
        for l in self.layers:
            num_units = l.compile(num_units)

        self.compiled = True

    def fit(self, X, y, num_iters):
        m = y.shape[0]
        for i in range(num_iters):
            A_in = X
            for l in self.layers:
                A_in = l.forward(A_in)

            loss = np.sum((A_in - y)**2) / (2 * m)
            grad = (A_in - y) / m
            for l in reversed(self.layers):
                grad = l.backward(grad)

            if i % 100 == 0:
                print(f"[{i}] Loss = {loss}")


    def __repr__(self):
        layers_repr = ""
        for l in self.layers:
            layers_repr += f"\t{l},\n"
        return f"Model ({len(self.layers)} layers, compiled={self.compiled})\n[{layers_repr}]"



X = np.random.random((10, 4))
y = np.random.random((10,1))
model = Model([ Input(units=4), Dense(units=12), ReLu(), Dense(units=6), ReLu(), Dense(units=1)])
model.compile()
print(model)

model.fit(X, y, num_iters=1000)