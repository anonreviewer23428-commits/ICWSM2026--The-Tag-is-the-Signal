# Codebook Tagger (QLoRA SFT)

This folder contains the training/inference utilities used for the paper’s codebook tagger (trained on 6×H200).

- Train: `python tagger/train/train_qlora.py --model_name <HF_OR_LOCAL_MODEL> --data_path <train_dataset_all.json> --output_dir <out_dir>`
- Evaluate agreement: `python tagger/utils/test.py --input <evaluation_results.json> --output <final_processed_results.json>`

Notes:
- Training requires a GPU environment (BitsAndBytes/FlashAttention/etc).
- The training data JSONs referenced in the paper (`train_dataset_all.json`, `train_dataset.json`) are not committed here.
