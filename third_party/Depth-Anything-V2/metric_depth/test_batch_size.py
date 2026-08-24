#!/usr/bin/env python3
"""
Test script pour vérifier la consommation mémoire GPU avec différents batch sizes
"""
import torch
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_anything_v2.dpt import DepthAnythingV2

def test_batch_size(batch_size, img_size=518):
    """Test la consommation mémoire pour un batch size donné"""
    print(f"\n=== Test batch size {batch_size} ===")
    
    # Configuration du modèle
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Créer le modèle
    model = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 41})
    model = model.cuda()
    model.eval()
    
    # Créer un batch fictif
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).cuda()
    
    # Mesurer la mémoire avant
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Mesurer la mémoire après
    mem_after = torch.cuda.memory_allocated() / 1024**3
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"  Mémoire avant forward: {mem_before:.2f} GB")
    print(f"  Mémoire après forward: {mem_after:.2f} GB")
    print(f"  Pic de mémoire: {mem_peak:.2f} GB")
    print(f"  Output shape: {output.shape}")
    
    # Nettoyage
    del model, dummy_input, output
    torch.cuda.empty_cache()
    
    return mem_peak

if __name__ == "__main__":
    print("Test de consommation mémoire GPU pour DepthAnythingV2 (ViT-Large)")
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    batch_sizes = [4, 8, 12, 16]
    results = {}
    
    for bs in batch_sizes:
        try:
            peak_mem = test_batch_size(bs)
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
        print(f"Batch size recommandé: {max_bs} (ou {max_bs-2} pour une marge de sécurité)")
