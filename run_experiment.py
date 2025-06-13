# run_experiment.py

import torch
import argparse
import os
import yaml
from src.config import get_config
from src.model import CST_VT
from src.dataset import create_dataloader
from src.trainer import Trainer, elbo_loss_fn
from src.utils import get_optimizer, get_scheduler

def main(args):
    # --- Load Configuration ---
    config = get_config()
    config['lead_time'] = args.lead_time
    print("--- Experiment Configuration ---")
    print(yaml.dump(config, default_flow_style=False))
    
    # --- Setup ---
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # --- Initialize Model ---
    model = CST_VT(
        grid_shape=config['data_params']['grid_shape'],
        patch_size=config['model_params']['patch_size'],
        in_channels=config['data_params']['num_variables'],
        embed_dim=config['model_params']['embed_dim'],
        depth=config['model_params']['num_layers'],
        num_heads=config['model_params']['num_heads'],
        num_input_time_steps=config['data_params']['time_steps'],
        causal_graph_path=config['data_paths']['causal_graph']
    )
    
    if args.mode == 'train':
        # --- Pre-training Stage ---
        print("\n--- Starting Pre-training Stage ---")
        pretrain_cfg = config['training_params']['pretrain']
        pretrain_loader = create_dataloader(
            data_path=config['data_paths']['cmip6_pretrain'],
            variables=config['data_params']['variables'],
            input_time_steps=config['data_params']['time_steps'],
            lead_time=config['lead_time'],
            batch_size=pretrain_cfg['batch_size']
        )
        
        optimizer = get_optimizer(model, pretrain_cfg)
        scheduler = get_scheduler(optimizer, pretrain_cfg, len(pretrain_loader) * pretrain_cfg['num_epochs'])
        trainer = Trainer(model, optimizer, scheduler, elbo_loss_fn, device, config)
        
        for epoch in range(pretrain_cfg['num_epochs']):
            print(f"Pre-train Epoch {epoch+1}/{pretrain_cfg['num_epochs']}")
            train_loss = trainer.train_one_epoch(pretrain_loader)
            print(f"Pre-train Loss: {train_loss:.4f}")

        # --- Fine-tuning Stage ---
        print("\n--- Starting Fine-tuning Stage ---")
        finetune_cfg = config['training_params']['finetune']
        finetune_loader = create_dataloader(
            data_path=config['data_paths']['soda_finetune'],
            variables=config['data_params']['variables'],
            input_time_steps=config['data_params']['time_steps'],
            lead_time=config['lead_time'],
            batch_size=finetune_cfg['batch_size']
        )
        # Validation loader to find the best model during fine-tuning
        val_loader = create_dataloader(
            data_path=config['data_paths']['godas_validate'],
            variables=config['data_params']['variables'],
            input_time_steps=config['data_params']['time_steps'],
            lead_time=config['lead_time'],
            batch_size=finetune_cfg['batch_size'],
            shuffle=False
        )

        optimizer = get_optimizer(model, finetune_cfg)
        scheduler = get_scheduler(optimizer, finetune_cfg, len(finetune_loader) * finetune_cfg['num_epochs'])
        trainer = Trainer(model, optimizer, scheduler, elbo_loss_fn, device, config)
        
        best_acc = -1.0
        for epoch in range(finetune_cfg['num_epochs']):
            print(f"Fine-tune Epoch {epoch+1}/{finetune_cfg['num_epochs']}")
            train_loss = trainer.train_one_epoch(finetune_loader)
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"Fine-tune Loss: {train_loss:.4f} | Val ACC: {val_metrics['acc']:.3f} | Val RMSE: {val_metrics['rmse']:.3f} | Val CRPS: {val_metrics['crps']:.3f}")

            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                checkpoint_path = os.path.join(config['checkpoint_dir'], f"cst_vt_lead{args.lead_time}_best.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best model saved to {checkpoint_path} with ACC: {best_acc:.3f}")

    elif args.mode == 'evaluate':
        # --- Evaluation Stage ---
        if not args.checkpoint_path:
            raise ValueError("Must provide --checkpoint_path for evaluation mode.")
        print(f"\n--- Starting Evaluation on Checkpoint: {args.checkpoint_path} ---")
        
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        
        val_loader = create_dataloader(
            data_path=config['data_paths']['godas_validate'],
            variables=config['data_params']['variables'],
            input_time_steps=config['data_params']['time_steps'],
            lead_time=config['lead_time'],
            batch_size=config['training_params']['finetune']['batch_size'],
            shuffle=False
        )
        
        trainer = Trainer(model, None, None, elbo_loss_fn, device, config)
        val_metrics = trainer.evaluate(val_loader)
        
        print("\n--- Evaluation Results ---")
        print(f"Lead Time: {args.lead_time} months")
        print(f"ACC:  {val_metrics['acc']:.4f}")
        print(f"RMSE: {val_metrics['rmse']:.4f}")
        print(f"CRPS: {val_metrics['crps']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CST-VT ENSO Forecasting Experiment")
    parser.add_argument('--lead_time', type=int, required=True, help="Forecast lead time in months.")
    parser.add_-argument('--mode', type=str, choices=['train', 'evaluate'], default='train', help="Mode to run: 'train' or 'evaluate'.")
    parser.add_argument('--checkpoint_path', type=str, help="Path to model checkpoint for evaluation.")
    
    args = parser.parse_args()
    main(args)
