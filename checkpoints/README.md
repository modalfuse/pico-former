# Trained checkpoints

The three Pico-Former models of the revised manuscript, trained on the AZT1D training split with seeds 20260828, 20260829, and 20260830 (RICE-Former v1.0.6 protocol and locked splits). Seed 20260828 is the deployed model.

| File | Seed | SHA-256 |
| --- | --- | --- |
| `s20260828/ceqt.pt` | 20260828 (deployed) | `18175b249372da8e50338ddcbcc5482d1a6171ecc219c2fc8114ee3f90f35ba8` |
| `s20260829/ceqt.pt` | 20260829 | `801ef1cf50e8ba278d9baec2ea523f688a42f95c7c29279acd28d1518f67c8f9` |
| `s20260830/ceqt.pt` | 20260830 | `fd696e298552cb685ea362271897a370262bcd25d55d8d89ef960a717119d02d` |

Each file is a PyTorch dictionary with the keys `state` (state dict of the network), `y_mean` and `y_std` (glucose normalization of the training split), `val` (validation score used for model selection), and `thresholds` (the 16 validation-frozen event thresholds \(\tau\)). It holds no CGM records or participant windows.

```python
import torch
ckpt = torch.load("checkpoints/s20260828/ceqt.pt", map_location="cpu", weights_only=True)
```

Set `RICE_CKPT` to one of these files to export the firmware weight header with `host/export_golden.py` or to run the host references in `code/`.

## License and attribution

The checkpoints are released under CC BY 4.0 (`licenses/CC-BY-4.0.txt`). They are derived from AZT1D; cite the dataset when using them:

> S. Khamesian, A. Arefeen, B. M. Thompson, M. A. Grando, and H. Ghasemzadeh, "AZT1D: A real-world dataset for type 1 diabetes," Mendeley Data, V1, 2025. https://doi.org/10.17632/gk9m674wcx.1
