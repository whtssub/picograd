import numpy as np
from picograd.engine import Value, optimize_graph, visualize_graph 

def test_sequential_backprop():
    try:
        print("Creating initial values...")
        a = Value(np.random.randn(10, 10))
        b = Value(np.random.randn(10, 10))
        c = Value(np.random.randn(10, 10))
        print("Initial values created successfully.")

        print("Performing forward pass...")
        d = a * b
        e = d + c
        f = e.relu()
        print("Forward pass completed successfully.")

        print("Performing backward pass...")
        f.backward()
        print("Backward pass completed successfully.")

        print(f"a.grad shape: {a.grad.data.shape}")
        print(f"b.grad shape: {b.grad.data.shape}")
        print(f"c.grad shape: {c.grad.data.shape}")

        print("Verifying gradients...")
        epsilon = 1e-6
        h = epsilon * np.ones_like(a.data.data)
        
        a_plus = Value(a.data.data + h)
        d_plus = a_plus * b
        e_plus = d_plus + c
        f_plus = e_plus.relu()
        
        numerical_grad = (f_plus.data.data - f.data.data) / epsilon
        
        max_diff = np.max(np.abs(numerical_grad - a.grad.data))
        print(f"Max difference between numerical and backprop gradients: {max_diff}")
        
        if max_diff < 1e-5:
            print("Gradient verification successful.")
        else:
            print("Warning: Large difference in gradients detected.")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()


def test_optimized_backprop():
    # Create initial values
    print("Creating initial values...")
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = Value(10.0, label='c')

    # Perform forward pass
    print("Performing forward pass...")
    d = a * b
    e = d + c
    f = e.relu()
    h = b.exp()
    i = e.log()

    # Visualize initial graph
    print("Visualizing initial graph...")
    visualize_graph(f, 'initial_graph')

    # Optimize graph
    print("Optimizing graph...")
    optimized_f = optimize_graph(f)

    # Visualize optimized graph
    print("Visualizing optimized graph...")
    visualize_graph(optimized_f, 'optimized_graph')

    # Perform backward pass
    print("Performing backward pass...")
    optimized_f.backward()

    # Visualize graph after backward pass
    print("Visualizing graph after backward pass...")
    visualize_graph(optimized_f, 'backward_graph')

    # Verify gradients numerically (Fix Me!)
    # print("\nVerifying gradients numerically...")
    # epsilon = 1e-6

    # def compute_numerical_grad(node):
    #     orig_value = node.data.data.copy()
    #     node.data.data += epsilon
    #     f_plus = optimized_f.data.data.copy()
    #     node.data.data = orig_value - epsilon
    #     f_minus = optimized_f.data.data.copy()
    #     node.data.data = orig_value
    #     return (f_plus - f_minus) / (2 * epsilon)

    # for node, label in [(a, 'a'), (b, 'b'), (c, 'c')]:
    #     numerical_grad = compute_numerical_grad(node)
    #     print(f"{label} numerical gradient: {numerical_grad.item():.4f}")
    #     print(f"{label} computed gradient: {node.grad.data.item():.4f}")
    #     print(f"Difference: {abs(numerical_grad.item() - node.grad.data.item()):.4e}\n")

if __name__ == "__main__":
    # test_sequential_backprop()
    test_optimized_backprop()