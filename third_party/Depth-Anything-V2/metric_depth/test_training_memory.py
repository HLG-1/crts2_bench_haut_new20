#!/usr/bin/env python3
"""
Test script plus réaliste pour vérifier la consommation mémoire GPU pendant l'entraînement
(incluant backward pass et optimiseur)
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss

def test_training_batch_size(batch_size, img_size=518):
    """Test la consommation mémoire pendant l'entraînement"""
    print(f"\n=== Test batch size {batch_size} (entraînement) ===")
    
    # Configuration du modèle
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Créer le modèle et le mettre sur GPU
    model = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 41})
    model = model.cuda()
    model.train()
    
    # Optimiseur (comme dans train.py)
    optimizer = AdamW([
        {'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': 0.000005},
        {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': 0.000005 * 10.0}
    ], lr=0.000005, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Loss function
    criterion = SiLogLoss().cuda()
    
    # Créer des données fictives
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).cuda()
    dummy_depth = torch.rand(batch_size, img_size, img_size).cuda() * 41
    dummy_mask = torch.ones(batch_size, img_size, img_size).cuda().bool()
    
    # Mesurer la mémoire avant
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    
    # Simulation d'une étape d'entraînement complète
    optimizer.zero_grad()
    pred = model(dummy_input)
    loss = criterion(pred, dummy_depth, dummy_mask)
    loss.backward()
    optimizer.step()
    
    # Mesurer la mémoire après
    mem_after = torch.cuda.memory_allocated() / 1024**3
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"  Mémoire avant entraînement: {mem_before:.2f} GB")
    print(f"  Mémoire après entraînement: {mem_after:.2f} GB")
    print(f"  Pic de mémoire: {mem_peak:.2f} GB")
    print(f"  Output shape: {pred.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Nettoyage
    del model, optimizer, criterion, dummy_input, dummy_depth, dummy_mask, pred, loss
    torch.cuda.empty_cache()
    
    return mem_peak

if __name__ == "__main__":
    print("Test de consommation mémoire GPU pour l'entraînement DepthAnythingV2 (ViT-Large)")
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    batch_sizes = [4, 8, 12, 16, 20, 24]
    results = {}
    
    for bs in batch_sizes:
        try:
            peak_mem = test_training_batch_size(bs)
            results[bs] = peak_mem
        except RuntimeError as e:
            print(f"  ❌ OOM avec batch size {bs}: {e}")
            results[bs] = None
            break
    
    print("\n=== Résumé ===")
    for bs, mem in results.items():
        if mem is not None:
            print(f"Batch size {bs}: {mem:.2f} GB (OK)")
        else:
            print(f"Batch size {bs}: OOM (échec)")
    
    # Recommandation
    print("\n=== Recommandation ===")
    successful_bs = [bs for bs, mem in results.items() if mem is not None]
    if successful_bs:
        max_bs = max(successful_bs)
        print(f"Batch size maximum testé: {max_bs}")
        safe_bs = max_bs - 2 if max_bs > 4 else max_bs
        print(f"Batch size recommandé: {safe_bs} (marge de sécurité)")
        print(f"Batch size agressif: {max_bs} (si vous voulez maximiser)")
