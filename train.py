import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from tqdm import tqdm

from models import FCGenerator, FCDiscriminator
from data.dataset_utils import get_mnist_loaders
from utils import load_config, set_seed


def setup_model(config):
    gen = FCGenerator(config).to(config.device)
    disc = FCDiscriminator(config).to(config.device)

    return gen, disc


def train(config): 
    set_seed(config.seed)
    G_model = FCGenerator(config).to(config.device)
    D_model = FCDiscriminator(config).to(config.device)
    criterion = torch.nn.BCELoss() 
    optimizer_G = torch.optim.Adam(G_model.parameters(), lr=config.lr_G, weight_decay=config.weight_decay)
    optimizer_D = torch.optim.Adam(D_model.parameters(), lr=config.lr_D, weight_decay=config.weight_decay)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

    train_dataloader, _, test_dataloader = get_mnist_loaders(config)

    logger = wandb.init(
        project="CGAN",
        # entity="petr-shuzlhenko",
        name=config.exp_name        
    )

    val_noise = torch.randn(10, config.z_dim).to(config.device)
    val_labels = F.one_hot(torch.tensor(np.arange(10)), num_classes=config.num_classes).float().to(config.device)

    for epoch in tqdm(range(config.num_epochs)):
        D_real_outputs = []
        D_fake_outputs = []
        D_losses = []
        G_losses = []

        for real_imgs, targets in train_dataloader:

            G_model.train()
            D_model.train()

            real_imgs_flatten = real_imgs.view(config.batch_size, -1).to(config.device)
            real_labels = torch.ones(config.batch_size, 1).to(config.device)
            fake_labels = torch.zeros(config.batch_size, 1).to(config.device)
            targets = F.one_hot(targets, num_classes=config.num_classes).float().to(config.device)

            # DISCRIMINATOR
            # REAL
            optimizer_D.zero_grad()

            outputs_real = D_model(real_imgs_flatten, targets)
            loss_real = criterion(outputs_real, real_labels)
            loss_real.backward()

            current_real_disc_output = outputs_real.mean().item()
            D_real_outputs.append(current_real_disc_output)
            
            # FAKE

            z = torch.randn(config.batch_size, config.z_dim).to(config.device)
            fake_imgs = G_model(z, targets)
            fake_imgs = fake_imgs.view(config.batch_size, -1)
            outputs_fake = D_model(fake_imgs.detach(), targets.detach())
            loss_fake = criterion(outputs_fake, fake_labels)
            loss_fake.backward()
            current_fake_disc_output = outputs_fake.mean().item()
            D_fake_outputs.append(current_fake_disc_output)

            D_loss = loss_real + loss_fake
            D_losses.append(D_loss.item())
            optimizer_D.step()

            # GENERATOR
            optimizer_G.zero_grad()
            outputs_d = D_model(fake_imgs, targets)
            loss_G = criterion(outputs_d, real_labels)
            G_losses.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()
        
        scheduler_G.step()
        scheduler_D.step()

        logger.log({
            'D_real_outputs': np.mean(D_real_outputs),
            'D_fake_outputs': np.mean(D_fake_outputs),
            'D_losses': np.mean(D_losses),
            'G_losses': np.mean(G_losses)
        })

        G_model.eval()
        with torch.no_grad():
            val_fake_imgs = G_model(val_noise, val_labels)
            val_fake_imgs = val_fake_imgs.view(-1, 28, 28).detach().cpu()
            # grid = torchvision.utils.make_grid(val_fake_imgs, nrow=5, normalize=True, value_range=(-1, 1))
            # grid = grid.permute(1, 2, 0)
            for i, img in enumerate(val_fake_imgs):
                logger.log({f"Image lbl {i}": wandb.Image(img, caption=f"Fixed noise lbl {i}")})

    logger.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment args parser")
    parser.add_argument("-pc", "--path2config", type=str, 
                        help="The path to yaml config file")
    args = parser.parse_args()

    config = load_config(args.path2config)

    train(config)