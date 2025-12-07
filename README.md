# Masked Autoencoder Pretraining on Strong Lensing Images

End-to-end code to reproduce the experiments in `Masked_Autoencoder_Pretraining_on_Strong_Lensing_Images_for_Joint_Dark_Matter_Model_Classification_and_Super_Resolution (2).pdf`:
- Phase 1: MAE pretraining on `no_sub` images.
- Phase 2: 3-way dark matter classification (axion, cdm, no_sub).
- Phase 3: LR -> HR super-resolution.
- Optional ablations (scratch vs pretrained, mask ratio study) and Optuna sweeps.

Repo highlights
- `mainv2.py`: single entry point for MAE + classification + SR, with ablations and Optuna.
- `analyze_ablation.py`: reads ablation CSVs and plots summaries.
- `outputs_lens/`: example artifacts (CSV metrics, pretrained weights, plots/reports).
- `Masked_Autoencoder_Pretraining_on_Strong_Lensing_Images_for_Joint_Dark_Matter_Model_Classification_and_Super_Resolution (2).pdf`: paper.

## Environment
- Python 3.9+ recommended; CUDA GPU strongly preferred but CPU works (slower).
- Core deps: torch, torchvision (for CUDA build use pip wheels), numpy, scikit-learn, scikit-image, matplotlib, tqdm, optuna, timm (optional import only).
- Example install (pick your CUDA wheel):
	```bash
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # or cpu wheels
	pip install numpy scikit-learn scikit-image matplotlib tqdm optuna timm
	```

## Data (manual download and placement)
Two archives are needed; download manually and rename exactly as below, no CLI downloads used here.
- Dataset1 (classification & MAE): https://drive.google.com/file/d/1znqUeFzYz-DeAE3dYXD17qoMPK82Whji/view?usp=sharing -> save as `Dataset1.zip`.
- Dataset2 (super-resolution): https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view?usp=sharing -> save as `Dataset2.zip`.

Manual placement checklist
1) Choose a working directory for data (e.g., `/workspaces/mae-lensing/data`).
2) Put `Dataset1.zip` and `Dataset2.zip` there.
3) Do **not** unpack yourself unless desired; `mainv2.py` auto-extracts if the zips exist. If already extracted, ensure the structure is:
	 ```
	 <data_root>/
		 axion/
			 *.npy
		 cdm/
			 *.npy
		 no_sub/
			 *.npy
		 HR/
			 *.npy
		 LR/
			 *.npy
	 ```

## Quickstart: full pipeline
Run all three phases with defaults (MAE 10 ep, CLS 10 ep, SR 10 ep) and save outputs to `<data_root>/outputs`:
```bash
python mainv2.py --data_root <data_root>
```
Artifacts written under `outputs/`: `mae_encoder.pth`, `classifier.pth`, `sr_model.pth`, ROC/t-SNE/confusion/reliability plots, SR examples, and logs.

## Key options (selected)
- `--data_root`: where `Dataset1.zip`/`Dataset2.zip` sit; script `chdir`s into it.
- `--mae_epochs`, `--mae_lr`, `--mae_mask_ratio`: MAE phase controls (default mask 0.75).
- `--batch_size`: global batch size (default 64).
- `--use_optuna --optuna_trials_cls --optuna_trials_sr`: enable small hyperparameter sweeps.
- `--run_ablation`: run scratch vs pretrained classification and SR ablations.
- `--run_mask_ablation --mask_ratios 0.5,0.75,0.9`: MAE mask-ratio study; per-phase ablation epochs set via `--mask_ablation_*_epochs`.

Command examples
- Full pipeline, default hyperparams:
	```bash
	python mainv2.py --data_root <data_root>
	```
- With Optuna sweeps (5 trials each) and ablations:
	```bash
	python mainv2.py --data_root <data_root> --use_optuna --optuna_trials_cls 5 --optuna_trials_sr 5 --run_ablation
	```
- Mask-ratio ablation only (shortened epochs):
	```bash
	python mainv2.py --data_root <data_root> --run_mask_ablation --mask_ratios 0.5,0.75,0.9 --mask_ablation_mae_epochs 5 --mask_ablation_cls_epochs 5 --mask_ablation_sr_epochs 5
	```

## Re-plot provided ablations
To summarize the bundled CSVs in `outputs_lens/` and regenerate plots:
```bash
python analyze_ablation.py --results_dir outputs_lens
```
Plots are saved to `outputs/` by default.

## Reference results (from `outputs_lens/`)
- Classification ablation (`ablation_cls.csv`): pretrained_full AUC 0.923 / ACC 0.672, pretrained_frozen AUC 0.537, scratch_full AUC 0.957 / ACC 0.825.
- Super-resolution ablation (`ablation_sr.csv`): pretrained_sr PSNR 33.05 / SSIM 0.9610; scratch_sr PSNR 33.01 / SSIM 0.9552.
- Mask-ratio study (`mask_ratio_ablation.csv`): mask 0.5 -> MAE loss 0.00255, CLS AUC 0.950, PSNR 34.01; mask 0.75 -> CLS AUC 0.950, PSNR 32.99; mask 0.9 -> CLS AUC 0.968, PSNR 32.65.
- Pretrained classifier report (`cls_pretrained_classification_report.txt`): overall ACC 0.8835; F1 axion 0.854, cdm 0.846, no_sub 0.953.

## Reproducibility notes
- Seeds: default 42; set via `--seed` for deterministic dataloading and torch configs.
- Image handling: `.npy` files loaded with `allow_pickle=True`, normalized to [0,1], resized to 64x64.
- Hardware: designed for GPU; CPU will work but much slower. Batch size 64 assumed; reduce if OOM.
- Outputs: all checkpoints and plots go to `<data_root>/outputs/`. Ablation CSVs/plots saved there unless you pass `--results_dir` to `analyze_ablation.py`.

## Paper
See `Masked_Autoencoder_Pretraining_on_Strong_Lensing_Images_for_Joint_Dark_Matter_Model_Classification_and_Super_Resolution (2).pdf` for full methodology and experiment design.