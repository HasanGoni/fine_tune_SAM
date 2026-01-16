# SAM3 Fine-Tuning Quick Start Guide

This directory contains everything you need to fine-tune SAM3 on your custom dataset.

## Prerequisites

1. **SAM3 Repository**: Clone the official repository
   ```bash
   cd /workspace
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e ".[train,dev]"
   ```

2. **Model Checkpoint**: Request access and download from HuggingFace
   - Visit: https://huggingface.co/facebook/sam3
   - Request access to the model
   - Login: `huggingface-cli login`

3. **Training Data**: Generate using the auto-label pipeline
   ```bash
   # Run notebook: nbs/06_sam3_autolabel_pipeline.ipynb
   # Set environment: RUN_SAM3=1
   ```

## Quick Start

### 1. Verify Your Setup

```bash
# Check if SAM3 is installed
python -c "import sam3; print('SAM3 installed successfully')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check training data
ls -la /workspace/data/autolabel/
```

### 2. Review Configuration

The training configuration is located at:
```
configs/sam3_custom_finetune.yaml
```

Key parameters you can modify:
- `batch_size`: Adjust based on your GPU memory (default: 4)
- `num_epochs`: Number of training epochs (default: 10)
- `learning_rate`: Base learning rate (default: 8e-5)
- `num_gpus`: Number of GPUs to use (default: 1)

### 3. Launch Training

**Single GPU:**
```bash
python training/train_sam3.py \
    --config configs/sam3_custom_finetune.yaml \
    --num-gpus 1
```

**Multi-GPU (4 GPUs):**
```bash
python training/train_sam3.py \
    --config configs/sam3_custom_finetune.yaml \
    --num-gpus 4
```

**Evaluation Mode:**
```bash
python training/train_sam3.py \
    --config configs/sam3_custom_finetune.yaml \
    --num-gpus 1 \
    --mode val
```

### 4. Monitor Training

Training outputs will be saved to:
```
./outputs/sam3_finetune/
├── checkpoints/
├── logs/
└── tensorboard/
```

View training logs:
```bash
tail -f outputs/sam3_finetune/logs/train.log
```

Launch TensorBoard:
```bash
tensorboard --logdir outputs/sam3_finetune/tensorboard
```

## Configuration Details

### Model Configuration
- Checkpoint: `facebook/sam3` (from HuggingFace)
- Segmentation enabled: Yes
- Interactive mode: No

### Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 8e-5 (transformer), 2.5e-5 (vision), 5e-6 (language)
- **Weight Decay**: 0.1
- **Gradient Clipping**: 0.1
- **Scheduler**: InverseSquareRoot with warmup
- **Warmup Steps**: 20
- **Resolution**: 1008

### Dataset
- Format: COCO
- Images: `/workspace/data/autolabel/images/`
- Annotations: `/workspace/data/autolabel/autolabel_annotations.json`

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config (try 2 or 1)
- Reduce `resolution` (try 512)
- Use gradient accumulation

### HuggingFace Access Error
- Ensure you've requested access to `facebook/sam3`
- Login: `huggingface-cli login`
- Wait for approval (usually within 24 hours)

### Import Error: No module named 'sam3'
- Install SAM3: `cd /workspace/sam3 && pip install -e ".[train,dev]"`
- Check installation: `pip show sam3`

### No Training Data
- Run the auto-label pipeline: `nbs/06_sam3_autolabel_pipeline.ipynb`
- Set `RUN_SAM3=1` to use real SAM3 model
- Verify data: `ls -la /workspace/data/autolabel/`

## Advanced Options

### Custom Dataset
Modify the config to point to your custom dataset:
```yaml
dataset:
  root: "/path/to/your/data"
  annotation_file: "/path/to/your/annotations.json"
  image_dir: "/path/to/your/images"
```

### Resume Training
Add to config:
```yaml
training:
  resume_from: "./outputs/sam3_finetune/checkpoints/checkpoint_1000.pth"
```

### Enable Weights & Biases
Modify config:
```yaml
logging:
  wandb:
    enabled: true
    project: "sam3-finetune"
    entity: "your-username"
```

## Next Steps

After training completes:

1. **Evaluate** your model on validation data
2. **Export** the best checkpoint
3. **Run inference** on new images
4. **Iterate** by adjusting hyperparameters

## Resources

- SAM3 GitHub: https://github.com/facebookresearch/sam3
- SAM3 Paper: https://arxiv.org/abs/2511.16719
- HuggingFace: https://huggingface.co/facebook/sam3
- Documentation: See `07_sam3_training_documentation.ipynb`
