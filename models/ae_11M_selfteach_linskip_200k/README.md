# ae_11M_selfteach_linskip_200k

From run `L1_11M_linskip`, `ckpt_200000.pt` (step 200000). Self-teaching AE, encoder 5,684,944 + decoder 5,685,616 params; torch 2.10.0a0+a36e1d39eb.nv26.01.42222806.

- `encoder.pt2`: images (B, 3, 96, 96) in [0, 1] -> z (B, 32, 32)
- `decoder.pt2`: z (B, 32, 32) -> images (B, 3, 96, 96) in [0, 1]

Exported in eval mode, fp32, dynamic batch 1-8192, like `export_refined_ae`. Load with
`torch.export.load(ae_dir / "encoder.pt2").module().to(device)`. Call it in fp32, outside
autocast: an exported program has fixed dtypes.

Test MSE through the reloaded programs (1024 STL10 test images, fp32): 0.00467.
