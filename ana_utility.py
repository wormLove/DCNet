import torch
import matplotlib.pyplot as plt

def visualize_tensor(tensor: torch.Tensor):
    """
    Visualize a 1D tensor and print the proportion of non-zero values.
    
    Args:
        tensor: A 1D torch.Tensor
    
    Returns:
        None
    """
    # Ensure the tensor is either [1, N] or [N]
    assert tensor.dim() == 2 and tensor.size(0) == 1 or tensor.dim() == 1, "Input tensor must be 1D or of shape [1, N]"
    
    # Flatten the tensor if necessary
    if tensor.dim() == 2:
        tensor = tensor.flatten()
    
    # Convert tensor to numpy array for plotting
    tensor_np = tensor.cpu().numpy()
    
    # Plot the tensor values using a bar plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tensor_np)), tensor_np)
    plt.title('1D Tensor Visualization')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
    # Calculate the proportion of non-zero values
    non_zero_count = torch.sum(tensor != 0).item()
    total_count = tensor.numel()
    non_zero_ratio = non_zero_count / total_count
    
    print(f'Proportion of non-zero values: {non_zero_ratio:.2f} ({non_zero_count}/{total_count})')
