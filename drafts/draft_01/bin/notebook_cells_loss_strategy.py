# =============================================================================
# CELL: Global Variables (replace existing globals cell)
# =============================================================================

# Set global variables

SEED = 42

DATASET_PATH = "../../dataset/"

BATCH_SIZE = 128
LATENT_DIM = 100
NUM_CLASSES = 10

NUM_STEPS = 15005
SAVE_INTERVAL = 1000

MODEL_OUTPUT_PATH = "model/"
CGAN_MODEL_NAME = "CGAN_DRAFT_01"
D_MODEL_NAME = "D_DRAFT_01"
G_MODEL_NAME = "G_DRAFT_01"

NUM_EVAL_SAMPLES = 10000

# Live monitor settings
LIVE_MONITOR = True
EMIT_INTERVAL = 100

# Loss function strategy: "bce", "lsgan", "hinge", "wgan"
LOSS_STRATEGY = "bce"


# =============================================================================
# CELL: Loss Strategies (new cell — insert after imports, before models)
# =============================================================================

import torch.nn.functional as F
from abc import ABC, abstractmethod


class GANLossStrategy(ABC):
    """Base class for GAN loss strategies."""
    use_sigmoid: bool = True
    
    @abstractmethod
    def d_loss_real(self, output):
        """Discriminator loss for real images."""
        pass
    
    @abstractmethod
    def d_loss_fake(self, output):
        """Discriminator loss for fake images."""
        pass
    
    @abstractmethod
    def g_loss(self, output):
        """Generator loss."""
        pass


class BCELossStrategy(GANLossStrategy):
    """Binary Cross-Entropy loss (original GAN)."""
    use_sigmoid = True
    
    def __init__(self, device):
        self.device = device
        self.criterion = nn.BCELoss()
        self._ones = None
        self._zeros = None
    
    def _ensure_labels(self, batch_size):
        if self._ones is None or self._ones.size(0) != batch_size:
            self._ones = torch.ones(batch_size, 1, device=self.device)
            self._zeros = torch.zeros(batch_size, 1, device=self.device)
    
    def d_loss_real(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._ones)
    
    def d_loss_fake(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._zeros)
    
    def g_loss(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._ones)


class LSGANLossStrategy(GANLossStrategy):
    """Least Squares loss — often more stable than BCE."""
    use_sigmoid = False
    
    def __init__(self, device):
        self.device = device
        self.criterion = nn.MSELoss()
        self._ones = None
        self._zeros = None
    
    def _ensure_labels(self, batch_size):
        if self._ones is None or self._ones.size(0) != batch_size:
            self._ones = torch.ones(batch_size, 1, device=self.device)
            self._zeros = torch.zeros(batch_size, 1, device=self.device)
    
    def d_loss_real(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._ones)
    
    def d_loss_fake(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._zeros)
    
    def g_loss(self, output):
        self._ensure_labels(output.size(0))
        return self.criterion(output, self._ones)


class HingeLossStrategy(GANLossStrategy):
    """Hinge loss — used in SAGAN, BigGAN."""
    use_sigmoid = False
    
    def d_loss_real(self, output):
        return torch.mean(F.relu(1.0 - output))
    
    def d_loss_fake(self, output):
        return torch.mean(F.relu(1.0 + output))
    
    def g_loss(self, output):
        return -torch.mean(output)


class WGANLossStrategy(GANLossStrategy):
    """Wasserstein loss — requires weight clipping or gradient penalty."""
    use_sigmoid = False
    
    def d_loss_real(self, output):
        return -torch.mean(output)
    
    def d_loss_fake(self, output):
        return torch.mean(output)
    
    def g_loss(self, output):
        return -torch.mean(output)


def get_loss_strategy(name: str, device) -> GANLossStrategy:
    """Factory function to get loss strategy by name."""
    strategies = {
        "bce": BCELossStrategy,
        "lsgan": LSGANLossStrategy,
        "hinge": HingeLossStrategy,
        "wgan": WGANLossStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown loss strategy: {name}. Options: {list(strategies.keys())}")
    
    # BCE and LSGAN need device for label tensors
    if name in ("bce", "lsgan"):
        return strategies[name](device)
    return strategies[name]()


# =============================================================================
# CELL: Discriminator Model (replace existing discriminator cell)
# =============================================================================

class Discriminator(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, use_sigmoid=True):
        
        super().__init__()
        
        self.use_sigmoid = use_sigmoid

        # Embed the class label into a vector of size 28*28 (one full image channel)
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)

        # Main sequential network that classifies the concatenated input
        self.model = nn.Sequential(
            # Input is (2, 28, 28): image channel + label channel
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            # Second conv block: (32, 14, 14) → (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            # Flatten: (64, 7, 7) → (64*7*7) = (3136)
            nn.Flatten(),

            # Dense(512, activation='relu')
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Output: single value (sigmoid applied conditionally)
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Embed label to 784 dims, reshape to a spatial map (1, 28, 28)
        label_embed = self.label_embedding(labels).squeeze(1)
        label_embed = label_embed.view(-1, 1, 28, 28)

        # Concatenate image and label map along channel axis
        x = torch.cat([img, label_embed], dim=1)

        x = self.model(x)
        
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        
        return x


# =============================================================================
# CELL: CGAN Assembly and Optimizers (replace existing cell)
# =============================================================================

# Get loss strategy
loss_strategy = get_loss_strategy(LOSS_STRATEGY, device)
print(f"Using loss strategy: {LOSS_STRATEGY} (sigmoid: {loss_strategy.use_sigmoid})")

# Instantiate models
g_model = Generator().to(device)
d_model = Discriminator(use_sigmoid=loss_strategy.use_sigmoid).to(device)

# Optimizers
optimizer_g = optim.Adam(g_model.parameters(), lr=0.001)
optimizer_d = optim.Adam(d_model.parameters(), lr=0.001)


# =============================================================================
# CELL: Training (replace existing training cell)
# =============================================================================

def plot_image(images, labels, rows, cols):
    """Plots a grid of generated images with their class labels."""
    fig = plt.figure(figsize=(8, 8))
    for i in range(rows * cols):
        img = images[i].detach().cpu().numpy().reshape(28, 28)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(str(labels[i].item()))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def train_model():
    # Fixed test noise and labels — used every SAVE_INTERVAL to visualize progress
    samples_test = torch.randn(16, LATENT_DIM, device=device)
    labels_test = torch.randint(0, 10, (16, 1), device=device)

    losses = {"G": [], "D": []}

    # Create an infinite iterator over the DataLoader
    data_iter = iter(train_loader)

    for step in range(NUM_STEPS):

        # --- Get a real batch from DataLoader ---
        try:
            real_imgs, batch_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_imgs, batch_labels = next(data_iter)

        real_imgs = real_imgs.to(device)
        batch_labels = batch_labels.unsqueeze(1).to(device)

        # --- Generate fake images ---
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        fake_imgs = g_model(noise, batch_labels)

        # --- Train Discriminator ---
        optimizer_d.zero_grad()

        d_real_out = d_model(real_imgs, batch_labels)
        d_loss_real = loss_strategy.d_loss_real(d_real_out)

        d_fake_out = d_model(fake_imgs.detach(), batch_labels)
        d_loss_fake = loss_strategy.d_loss_fake(d_fake_out)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        optimizer_d.step()

        # --- Train Generator ---
        optimizer_g.zero_grad()

        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        gen_labels = torch.randint(0, 10, (BATCH_SIZE, 1), device=device)

        gen_imgs = g_model(z, gen_labels)
        g_out = d_model(gen_imgs, gen_labels)

        g_loss = loss_strategy.g_loss(g_out)
        g_loss.backward()
        optimizer_g.step()

        # --- Logging ---
        losses["G"].append(g_loss.item())
        losses["D"].append(d_loss.item())

        if step % SAVE_INTERVAL == 0:
            print(f"Step {step} — D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
            with torch.no_grad():
                results = g_model(samples_test, labels_test)
            plot_image(results, labels_test, 4, 4)

        if LIVE_MONITOR and step % EMIT_INTERVAL == 0:
            with torch.no_grad():
                results = g_model(samples_test, labels_test)
            emit_frames(results, labels_test, step, g_loss.item(), d_loss.item())

    return losses


losses = train_model()

if LIVE_MONITOR:
    emit_done()
