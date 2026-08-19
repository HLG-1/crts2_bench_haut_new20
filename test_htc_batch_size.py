#!/usr/bin/env python3
"""
Test script pour vérifier la consommation mémoire GPU avec HTC-DC-Net
"""
import sys
import os
import torch
import numpy as np

# Ajouter HTC-DC-Net au PYTHONPATH
htc_dir_abs = os.path.abspath("third_party/HTC-DC-Net")
if htc_dir_abs not in sys.path:
    sys.path.insert(0, htc_dir_abs)

from htcdc import UBins

def test_htc_batch_size(batch_size, image_size=256):
    """Test la consommation mémoire pour un batch size donné"""
    print(f"\n=== Test batch size {batch_size} (HTC-DC-Net) ===")
    
    # Configuration du modèle (similaire à configs/htcdc.yaml)
    cfgs = {
        "model": "htcdc",
        "backbone": "efficientnetb0",
        "patch_size": 4,
        "num_classes": 256,
        "fusion_mode": "last",
        "head_tail_cut": False,
        "earlier": False,
        "prob_loss": False,
        "data_dir": "data/htc_dc_net_format",
        "test": True,
        "device": "cuda",
    }
    
    try:
        # Créer le modèle
        model = UBins(cfgs)
        model = model.cuda()
        model.train()
        
        # Créer un batch fictif
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).cuda()
        dummy_gt = torch.randn(batch_size, 1, image_size, image_size).cuda()
        
        # Mesurer la mémoire avant
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        
        # Forward pass
        losses, pred = model(dummy_input, dummy_gt)
        
        # Mesurer la mémoire après forward
        mem_after_forward = torch.cuda.memory_allocated() / 1024**3
        mem_peak_forward = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"  Mémoire avant forward: {mem_before:.2f} GB")
        print(f"  Mémoire après forward: {mem_after_forward:.2f} GB")
        print(f"  Pic de mémoire (forward): {mem_peak_forward:.2f} GB")
        print(f"  Output shape: {pred[0].shape if isinstance(pred, list) else pred.shape}")
        
        # Test backward pass
        loss_total = losses["loss_total"]
        loss_total.backward()
        
        mem_after_backward = torch.cuda.memory_allocated() / 1024**3
        mem_peak_backward = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"  Mémoire après backward: {mem_after_backward:.2f} GB")
        print(f"  Pic de mémoire (total): {mem_peak_backward:.2f} GB")
        
        # Nettoyage
        del model, dummy_input, dummy_gt, losses, pred, loss_total
        torch.cuda.empty_cache()
        
        return mem_peak_backward
        
    except RuntimeError as e:
        print(f"  ❌ OOM avec batch size {batch_size}: {str(e)[:100]}")
        return None

if __name__ == "__main__":
    print("Test de consommation mémoire GPU pour HTC-DC-Net (EfficientNet-B0)")
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Tester différents batch sizes
    batch_sizes = [8, 16, 24, 32, 48, 64]
    results = {}
    
    for bs in batch_sizes:
        try:
            peak_mem = test_htc_batch_size(bs)
            results[bs] = peak_mem
            if peak_mem is None:
                break
        except Exception as e:
            print(f"  ❌ Erreur avec batch size {bs}: {e}")
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
        safe_bs = max_bs - 8 if max_bs > 16 else max_bs - 4
        print(f"Batch size recommandé: {safe_bs} (marge de sécurité)")
        print(f"Batch size agressif: {max_bs} (si vous voulez maximiser)")
