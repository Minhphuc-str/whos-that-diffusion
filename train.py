import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PokemonDataset
from networks import UNetGenerator, PatchGANDiscriminator, init_weights
import torchvision

# --- Configuration ---
NUM_EPOCHS = 200        # Standard for Pix2Pix to get good results
BATCH_SIZE = 4          # Keep small (1-4) for U-Net stability on small GPUs
LR = 0.0002             # Learning Rate
BETA1 = 0.5             # Adam Beta1 (Crucial for GANs to avoid mode collapse)
LAMBDA_L1 = 100         # How strictly we force the generator to match ground truth colors
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"

# Auto-detect Mac (MPS), NVIDIA (CUDA), or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def save_sample_images(generator, val_loader, epoch):
    """Saves a grid of (Input | Generated | Real) images to monitor progress"""
    generator.eval()
    with torch.no_grad():
        # Grab a single batch
        batch = next(iter(val_loader))
        real_A = batch['A'].to(DEVICE)
        real_B = batch['B'].to(DEVICE)
        
        # Generate fake image
        fake_B = generator(real_A)
        
        # Stack images horizontally: Input -> Generated -> Real
        # Denormalize from [-1, 1] to [0, 1] for saving
        img_sample = torch.cat([real_A, fake_B, real_B], dim=3)
        img_sample = (img_sample + 1) / 2.0
        
        save_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch}.png")
        torchvision.utils.save_image(img_sample, save_path, nrow=1)
    generator.train()

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print(f"Starting training on: {DEVICE}")

    # 1. Init Data
    dataset = PokemonDataset(root_dir='dataset', mode='train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. Init Models
    netG = UNetGenerator(input_nc=3, output_nc=3).to(DEVICE)
    netD = PatchGANDiscriminator(input_nc=6).to(DEVICE) # Input is 6 because D sees (Silhouette + Color)
    
    # Initialize weights (Gaussian distribution helps GAN convergence)
    init_weights(netG)
    init_weights(netD)

    # 3. Optimizers & Loss
    optimizer_G = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    criterionGAN = nn.BCEWithLogitsLoss() # Binary Cross Entropy (Real vs Fake)
    criterionL1 = nn.L1Loss()             # Pixel distance (Input vs Target)

    # 4. Training Loop
    print(f"Images per epoch: {len(dataset)}")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            # Unpack data
            real_A = batch['A'].to(DEVICE) # Input Silhouette
            real_B = batch['B'].to(DEVICE) # Ground Truth Color

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # 1. Real Loss: D(A + B) should be 1 (Real)
            # We concat A and B because D checks if the color MATCHES the shape
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            
            # 2. Fake Loss: D(A + Fake_B) should be 0 (Fake)
            fake_B = netG(real_A)
            fake_AB = torch.cat((real_A, fake_B.detach()), 1) # .detach() prevents G from updating here
            pred_fake = netD(fake_AB)
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            
            # Combine and Backprop
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # 1. GAN Loss: D(A + Fake_B) should be 1 (Try to fool D)
            fake_AB = torch.cat((real_A, fake_B), 1) # No detach! We want G to learn
            pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
            
            # 2. L1 Loss: Fake_B should look like Real_B (Ground Truth)
            loss_G_L1 = criterionL1(fake_B, real_B) * LAMBDA_L1
            
            # Combine
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # Logging
            if i % 10 == 0:
                print(f"\r[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(loader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]", end="")

        # End of Epoch
        print(f" - Time: {time.time() - start_time:.1f}s")
        
        # Save Visuals every 5 epochs
        if epoch % 5 == 0 or epoch == 0:
            save_sample_images(netG, loader, epoch)
            
        # Save Model Checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, f"netG_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(CHECKPOINT_DIR, f"netD_{epoch}.pth"))

    print("Training Finished!")
    torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, "netG_final.pth"))

if __name__ == "__main__":
    train()