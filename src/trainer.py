# src/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import AnomalyCorrelation, RootMeanSquaredError, ContinuousRankedProbabilityScore

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        
        # Evaluation metrics
        self.acc = AnomalyCorrelation()
        self.rmse = RootMeanSquaredError()
        self.crps = ContinuousRankedProbabilityScore()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Training")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            mu, log_var = self.model(inputs)
            loss = self.loss_fn(mu, log_var, targets.unsqueeze(1))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        if self.scheduler:
            self.scheduler.step()
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds_mu = []
        all_preds_std = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc="Evaluating")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            mu, log_var = self.model(inputs)
            loss = self.loss_fn(mu, log_var, targets.unsqueeze(1))
            
            total_loss += loss.item()
            
            all_preds_mu.append(mu.cpu())
            all_preds_std.append(torch.exp(0.5 * log_var).cpu())
            all_targets.append(targets.cpu())
            
        avg_loss = total_loss / len(dataloader)
        
        # Concatenate all batches
        preds_mu = torch.cat(all_preds_mu).squeeze()
        preds_std = torch.cat(all_preds_std).squeeze()
        targets = torch.cat(all_targets)
        
        # Calculate metrics
        acc_score = self.acc(preds_mu, targets)
        rmse_score = self.rmse(preds_mu, targets)
        crps_score = self.crps(preds_mu, preds_std, targets)
        
        return {
            "loss": avg_loss,
            "acc": acc_score.item(),
            "rmse": rmse_score.item(),
            "crps": crps_score.item()
        }

def elbo_loss_fn(mu, log_var, y, beta=0.1):
    """Evidence Lower Bound (ELBO) loss for a Gaussian likelihood."""
    # Reconstruction loss (Negative Log-Likelihood)
    recon_loss = F.gaussian_nll_loss(mu, y, torch.exp(log_var), reduction='mean')
    
    # KL divergence between q(z|x) and p(z)=N(0,I)
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kl_div
