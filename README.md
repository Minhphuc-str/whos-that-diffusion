# ❓ Who's That Diffusion?

**From "Who's That Pokémon?" to State-of-the-Art Generative Vision.**

This repository documents the development of a generative computer vision pipeline capable of hallucinating high-fidelity, Ken Sugimori-style Pokémon artwork from rough, hand-drawn silhouettes.

The project serves as a deep dive into the evolution of Image-to-Image translation, moving from classical adversarial networks (GANs) to modern latent diffusion models.

## 🎯 Project Goals
* **Abstract Generalization:** The model should not just memorize existing Pokémon shapes. If given a silhouette of a dog or a toaster, it should generate a plausible "Pokémon-ified" version of that object.
* **Robust Inference:** Handling imperfect, wobbly human sketches using synthetic training data augmentation (Elastic Deformations).
* **Architectural Comparison:** Benchmarking **Pix2Pix (cGAN)** against **ControlNet (Stable Diffusion Adapter)** to analyze the trade-off between control and creativity.

## 🛠️ Technical Stack
* **Core:** Python 3.10, PyTorch
* **Data Pipeline:** OpenCV, NumPy, Scipy (Elastic Deformations), DuckDuckGo Search
* **Architectures:**
    * **Baseline:** Pix2Pix (U-Net Generator + PatchGAN Discriminator)
    * **Advanced:** Stable Diffusion + ControlNet / LoRA
* **Environment:** Conda

## 📊 The Data Pipeline (`/data`)
To simulate the imperfection of human drawing during training, I developed a custom data generation pipeline. Instead of training on perfect binary masks (which leads to "texture stickiness" and poor generalization), the pipeline applies:

1.  **Automated Scraping:** Retrieving official artwork with transparency handling.
2.  **Elastic Deformation:** Applying randomized Gaussian filters to displacement fields (based on *Simard et al.*) to create organic, "wobbly" silhouettes.
3.  **Morphological Noise:** Random erosion/dilation to simulate varying pen thickness.

## 🧠 Model Architectures

### Phase 1: Conditional GAN (Pix2Pix)
*Implementation of the classic Image-to-Image translation paper.*
* **Objective:** Learn a mapping $G: X \to Y$ using an adversarial loss combined with an L1 reconstruction loss.
* **Why:** Establishes a baseline and tests the limits of non-latent generative models on stylized data.

### Phase 2: Latent Diffusion (ControlNet)
*Fine-tuning a pre-trained generative model.*
* **Objective:** Leverage the semantic knowledge of Stable Diffusion (which knows what "animals" look like) and constrain it using the silhouette spatial condition.
* **Why:** To achieve the "Abstract Generalization" goal. A GAN trained only on 151 Pokémon cannot imagine a "Pokémon Dog," but a Diffusion model pre-trained on LAION-5B can.

## 🚀 Usage

### 1. Environment Setup
We use `conda` for robust OpenCV dependency management and `pip` for the search tools.

```bash
conda create -n poke-gen python=3.10 -y
conda activate poke-gen
conda install -c conda-forge numpy opencv requests -y
pip install duckduckgo-search
