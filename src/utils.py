# src/utils.py

import torch

class AnomalyCorrelation(object):
    """Computes Anomaly Correlation Coefficient."""
    def __call__(self, preds, targets):
        preds_anom = preds - preds.mean()
        targets_anom = targets - targets.mean()
        
        numerator = (preds_anom * targets_anom).mean()
        denominator = torch.sqrt((preds_anom**2).mean() * (targets_anom**2).mean())
        
        return numerator / (denominator + 1e-6)

class RootMeanSquaredError(object):
    """Computes Root Mean Squared Error."""
    def __call__(self, preds, targets):
        return torch.sqrt(((preds - targets)**2).mean())

class ContinuousRankedProbabilityScore(object):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for a Gaussian prediction.
    """
    def __call__(self, mu, sigma, y):
        # mu, sigma, y should be 1D tensors
        y = y.view(-1, 1)
        mu = mu.view(-1, 1)
        sigma = sigma.view(-1, 1)
        
        # Standardize y
        y_std = (y - mu) / (sigma + 1e-8)
        
        # Standard normal PDF and CDF
        pdf = torch.exp(-0.5 * y_std**2) / (2 * np.pi)**0.5
        cdf = 0.5 * (1 + torch.erf(y_std / 2**0.5))
        
        crps = sigma * (y_std * (2 * cdf - 1) + 2 * pdf - 1 / np.pi**0.5)
        return crps.mean()

def get_optimizer(model, config):
    if config['optimizer'] == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported.")

def get_scheduler(optimizer, config, num_steps):
    if config['scheduler'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    elif config['scheduler'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return None
