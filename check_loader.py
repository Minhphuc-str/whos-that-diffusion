import torch
from torch.utils.data import DataLoader
from dataset import PokemonDataset
import matplotlib.pyplot as plt
import torchvision

def show_tensor_images(image_tensor, num_images=4):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2 # Denormalize from [-1, 1] to [0, 1]
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=4)
    plt.figure(figsize=(10,5))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Initialize Dataset
    dataset = PokemonDataset(root_dir='dataset', mode='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    real_A = batch['A'] # Silhouettes
    real_B = batch['B'] # Targets
    
    print(f"Batch Shape: {real_A.shape}")
    print("Showing Silhouettes (Input)...")
    show_tensor_images(real_A)
    
    print("Showing Targets (Ground Truth)...")
    show_tensor_images(real_B)