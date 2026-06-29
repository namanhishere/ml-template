# AGENTS.md — AI/ML/DL PyTorch Template

## Setup & Commands

```bash
# Install all deps (core + dev)
uv sync

# Install just core deps
uv sync --no-dev

# Lint + format
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/

# Typecheck
mypy src/

# Run all tests with coverage
pytest -v --cov=src

# Run a single test file
pytest tests/test_models.py -v

# Pre-commit hooks (run before committing)
pre-commit run --all-files
```

## Architecture

### Pipeline: Config → Model → Experiment → Trainer

```
Hydra config (configs/config.yaml)
  ↓ composes via defaults list
  ↓ selects: experiment, model, dataset, optimizer, scheduler, loss, trainer
  ↓
scripts/train.py
  ↓ Trainer.build_from_config(cfg)
  ↓   creates model (via MODELS registry or BackboneFactory)
  ↓   creates experiment (via EXPERIMENTS registry)
  ↓   creates callbacks (via CALLBACKS registry)
  ↓   creates datamodule
  ↓
trainer.fit()
  ↓   builds optimizer, scheduler, scaler
  ↓   train_epoch → experiment.forward → experiment.compute_loss
  ↓   validate_epoch → experiment.forward → experiment.compute_metrics
  ↓   callbacks: checkpoint save, early stop, EMA, MLflow log
```

**Never put training logic in notebooks.** Training is CLI-only via `scripts/train.py`. Notebooks are for EDA (`01_explore_data.ipynb`) and evaluation/error analysis (`02_evaluate_results.ipynb`) only.

## Registry Pattern (Central to Everything)

8 registries live in `src/utils/registry.py`: `DATASETS`, `MODELS`, `LOSSES`, `METRICS`, `EXPERIMENTS`, `CALLBACKS`, `PREDICTORS`, `FINETUNE_STRATEGIES`.

Components self-register via decorators at module import time:

```python
from src.utils.registry import LOSSES

@LOSSES.register("focal")
class FocalLoss(nn.Module):
    ...

# Must import the module to trigger registration:
from src.losses.losses.classification import FocalLoss  # now in LOSSES
```

**Critical**: registries are empty until the module containing the decorator is imported. Components are NOT auto-discovered. Each registry's `__init__.py` imports subordinate modules to trigger registration.

### Adding a new component

1. Create file in the right subdirectory
2. Decorate with `@REGISTRY.register("name")`
3. Import it in that directory's `__init__.py`
4. Add a Hydra config in `configs/<type>/`

## Config Structure (Hydra)

Main config at `configs/config.yaml` composes sub-configs via `defaults:` list. Keys are **nested** under their type, not flat:

```yaml
# configs/config.yaml
model:
  name: resnet50
trainer:
  batch_size: 32       # ← NOT config.batch_size, it's config.trainer.batch_size
  max_epochs: 100      # ← NOT config.max_epochs
  precision: fp16      # ← enables AMP GradScaler
```

Override anything on CLI:
```bash
python scripts/train.py model=vit trainer.batch_size=64 trainer.max_epochs=200
```

Hydra sweep:
```bash
python scripts/train.py --multirun optimizer.params.lr=0.001,0.01,0.1
```

## Domain Logic: Experiment Classes

**All domain logic lives in experiment classes**, NOT in trainer. Trainer delegates to experiment — no if-else chains for different task types.

| Experiment | `forward()` returns | `compute_loss()` uses |
|-----------|--------------------|------------------|
| `ClassificationExperiment` | `{"logits": ...}` | cross_entropy / focal |
| `SegmentationExperiment` | `{"logits": ...}` | dice + CE combined |
| `RegressionExperiment` | `{"values": ...}` | mse / mae / huber |

Each experiment defines: `forward()`, `compute_loss()`, `compute_metrics()`, `visualize()`, `postprocess()`.

To add a new task: create a new experiment class, register it, add Hydra config — no changes to trainer needed.

## Backbone Factory

Backbones use `source://name` format:
```python
from src.models.zoo import BackboneFactory
model = BackboneFactory.create("torchvision://resnet50", pretrained=True)
model = BackboneFactory.create("timm://efficientnet_b0")       # needs timm plugin
model = BackboneFactory.create("hf://bert-base-uncased")       # needs HF plugin
```

Built-in sources: `torchvision` (35+ models). Plugins in `src/plugins/` add `timm` and `hf` backends — they auto-register when `src/plugins/timm/__init__.py` is imported.

Feature dimensions for head construction:
```python
dim = BackboneFactory.get_feature_dim("torchvision://resnet50")  # → 2048
```

## Training

```bash
# Single GPU
python scripts/train.py trainer.batch_size=64

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train.py trainer.batch_size=32

# Resume from checkpoint
python scripts/train.py +ckpt_path=checkpoints/last.pt
```

The `setup_logging()` in `src/utils/distributed.py` auto-detects DDP state — ranks, world size, and `is_main_process()` are set at import time. Rank-0-only logging, checkpoint saving, and progress bars are enforced automatically.

## Checkpoints

Each checkpoint is a single `.pt` dict with 30+ keys: `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, `ema_state`, `epoch`, `global_step`, `dataloader_state`, `best_metric`, `metrics_history`, `last_metrics`, `config` (full Hydra snapshot), `seed`, `command`, `random_state`, `git_hash`, `framework_version`, `world_size`, `local_rank`, `dataset_version`.

Saving triggers (configurable):
| Trigger | Filename |
|---------|----------|
| Every N epochs | `epoch_{epoch:03d}.pt` |
| Best metric | `best.pt` |
| Latest epoch | `last.pt` |
| Interrupt | `interrupt.pt` |

## Inference & Serving

```bash
# Batch predict via Predictor
python scripts/predict.py checkpoints/best.pt --input image.jpg

# Start HTTP server
python scripts/serve.py --ckpt-path checkpoints/best.pt --port 8000
```

**Predictor is the single inference interface** — used by both `predict.py` and `serving/app.py`. No duplicated inference logic. The FastAPI app loads the model via `CKPT_PATH` environment variable on startup.

## Experimentation & Analysis Tools (`src/tools/`)

Before training or after checkpoints exist, use these to validate the pipeline and analyze results.

### Pipeline Validation

```bash
# Feasibility analysis: overfit test + random label test
python scripts/feasibility.py model=resnet50 tools.feasibility.tests=[overfit,random_label]

# Linear probe: freeze backbone, train linear classifier
python scripts/linear_probe.py model=resnet50
```

### Fine-Tuning

```bash
# Fine-tune with configurable strategy
python scripts/fine_tune.py tools.fine_tuning.strategy=differential tools.fine_tuning.lr_multiplier=0.1

# Available strategies (registered via FINETUNE_STRATEGIES registry):
#   full           — unfreeze everything
#   differential   — backbone gets lower LR
#   last_n         — unfreeze only last N layers  
#   head_only      — train only classifier head
#   gradual_unfreeze — unfreeze one layer every N epochs
```

### Data Analysis

```bash
# Few-shot evaluation: test performance with K samples per class
python scripts/few_shot.py tools.few_shot.k_shots=[1,5,10,20] tools.few_shot.mode=pretrained

# Data ablation: systematically remove data and measure impact
python scripts/data_ablation.py tools.data_ablation.strategies=[size,class,difficulty]
```

### Post-Training Analysis

```bash
# Learning curve analysis: auto-diagnose training behavior
python scripts/learning_curves.py checkpoints/last.pt
```

### Tool Architecture

| Tool | Source | Key Design |
|------|--------|------------|
| `FeasibilityAnalyzer` | `src/tools/feasibility.py` | `run(tests=["overfit", "random_label"])` — extensible, new tests added without API change |
| `LinearProbe` | `src/tools/linear_probe.py` | Validates/warns if backbone was not pretrained |
| `FineTuner` | `src/tools/fine_tuning.py` | `FineTuneStrategy` ABC + `FINETUNE_STRATEGIES` registry — extensible for LoRA, Adapters, PEFT |
| `FewShotEvaluator` | `src/tools/few_shot.py` | Subset-based K-shot only (no meta-learning); `mode: pretrained|scratch` |
| `DataAblator` | `src/tools/data_ablation.py` | Size/class/difficulty ablation + training-dynamics influence scores (forgetting, difficulty, confidence) |
| `LearningCurveAnalyzer` | `src/tools/learning_curves.py` | Auto-detects 6 conditions: overfitting, underfitting, divergence, plateau, high LR, low LR |
| `ReportGenerator` | `src/tools/reporting.py` | Self-contained Plotly HTML reports in `reports/experiment_NNN/index.html` |

**FineTuneStrategy extensibility**: to add LoRA later, create `@FINETUNE_STRATEGIES.register("lora")` class implementing `configure()`, import it — no changes to `FineTuner` needed.

**Reports**: generated as single `.html` files with embedded Plotly charts. Graceful fallback to JSON if plotly not installed.

## Key Constraints & Gotchas

1. **Config key nesting**: read `config.trainer.batch_size` not `config.batch_size`. The Hydra `config.yaml` nests everything under its type key.
2. **BaseDataModule reads top-level keys**: `config.get("batch_size")` — when called with the full config, it won't find keys nested under `trainer:`. The Trainer passes the whole config.
3. **Registries are pop-on-import**: `@REGISTRY.register("name")` runs at module import. If a component isn't appearing in a registry, check its module is imported in the directory `__init__.py`.
4. **Backbone format always `source://name`**: plain names like `"resnet50"` will fail.
5. **`src/utils/distributed.py` has module-level side effects**: it auto-detects DDP at import time via `_detect_distributed()`. RANK, WORLD_SIZE, IS_DISTRIBUTED are set once.
6. **Line length**: 120 (ruff config). **Quote style**: double quotes. **Indent**: spaces.
7. **Tests**: `pythonpath = ["."]` in pyproject.toml — imports use `from src.X import Y` from project root.
8. **Optional deps**: `pydantic`, `pandas`, `mlflow`, `onnx`, `timm`, `transformers`, `wandb`, `albumentations` are in `[project.optional-dependencies] dev` — they may not be installed.
