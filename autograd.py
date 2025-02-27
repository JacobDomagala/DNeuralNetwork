class Neuron:
    def __init__(self, data, children = ()):
        self.grad = 0.0
        self.data = data
        self.children_ = children
        self.backward_ = lambda : None

    def backward(self):
        visited = []
        nodes = []

        def build(node):
            if node not in visited:
                visited.append(node)
                for c in node.children_:
                    build(c)

                nodes.append(node)

        build(self)

        self.grad = 1.0
        for n in reversed(nodes):
            n.backward_()

    def __add__(self, other):
        other = other if isinstance(other, Neuron) else Neuron(data=other)
        out = Neuron(data=self.data + other.data, children=(self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_ = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Neuron) else Neuron(data=other)
        out = Neuron(data=self.data*other.data, children=(self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward_ = backward

        return out

    def relu(self):
        out = Neuron(data = max(0, self.data), children=(self,))

        def backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out.backward_ = backward
        return out

    def __repr__(self):
        return f"Neuron(data={self.data}, leaf={len(self.children_) == 0}, grad={self.grad})\n"


a = Neuron(data = 10.0)
b = Neuron(data = 2.0)
c = a + b
d = c * 3.0
e = d.relu()
f = e + 4.0
f.backward()

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)