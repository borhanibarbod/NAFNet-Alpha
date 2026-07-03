# Checkpoints

The final model weights are not stored in git. Download them from the GitHub Releases page of this repository and place the file here:

```
checkpoints/EXPB_FINAL_a0_42.60.pth
```

File details:

- Size: 81.8 MB
- Parameters: 20,379,939
- Format: PyTorch checkpoint, state dict stored under the `model_state` key
- Results (internal SIDD validation, 1,000 patches): 42.60 dB PSNR at alpha=0, 38.85 dB at alpha=0.5, 34.64 dB at alpha=1
