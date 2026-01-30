import random
from picograd.engine import Value, visualize_graph

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def generate_dataset(n):
    return [
        ([random.uniform(-1, 1), random.uniform(-1, 1)], Value(1))
        if random.random() > 0.5
        else ([random.uniform(-1, 1), random.uniform(-1, 1)], Value(-1))
        for _ in range(n)
    ]

def train_model():
    # Create the model
    model = MLP(2, [8, 8, 1])  # 2 inputs, two hidden layers of 8 neurons, 1 output
    print("Model created")

    # Generate a dataset
    data = generate_dataset(100)
    print("Dataset generated")

    # Visualize the initial model
    x_sample, _ = data[0]
    y_pred = model([Value(xi) for xi in x_sample])
    visualize_graph(y_pred, filename='initial_model')
    print("Initial model visualized")

    # Training loop
    learning_rate = 0.01
    for epoch in range(1000):
        # Forward pass
        total_loss = Value(0)
        for x, y in data:
            x = [Value(xi) for xi in x]
            y_pred = model(x)
            loss = (y_pred - y)**2
            total_loss = total_loss + loss
        
        # Backward pass
        for p in model.parameters():
            p.grad.data = 0
        total_loss.backward()
        
        # Update
        for p in model.parameters():
            p.data.data -= learning_rate * p.grad.data
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.data.data.item() / len(data):.4f}")
            
            # Debug information
            print(f"  Sample prediction: {y_pred.data.data.item():.6f}")
            print(f"  Sample target: {y.data.data.item():.6f}")
            
            # Check parameter values
            param_values = [p.data.data.item() for p in model.parameters()]
            print(f"  Parameter range: {min(param_values):.6f} to {max(param_values):.6f}")
            
            # Check gradients
            grad_values = [p.grad.data.item() for p in model.parameters()]
            print(f"  Gradient range: {min(grad_values):.6f} to {max(grad_values):.6f}")

    print("Training completed")

    # Visualize the trained model
    y_pred = model([Value(xi) for xi in x_sample])
    visualize_graph(y_pred, filename='trained_model')
    print("Trained model visualized")
    
    # Final prediction check
    print("\nFinal predictions:")
    for i in range(5):
        x, y = data[i]
        y_pred = model([Value(xi) for xi in x])
        print(f"Input: {x}, Target: {y.data.data.item():.6f}, Prediction: {y_pred.data.data.item():.6f}")

if __name__ == "__main__":
    train_model()