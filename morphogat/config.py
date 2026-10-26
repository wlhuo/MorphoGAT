"""Validated experiment configuration."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PreprocessConfig:
    min_genes: int = 700
    max_mito_fraction: float = 0.35
    min_gene_fraction: float = 0.025
    library_size: float = 10000.0
    clip: float = 10.0
    pca_components: int = 50
    patch_size: int = 64
    allow_partial_images: bool = False
    # Missing training-panel RNA features are rejected unless explicitly allowed.
    allow_missing_genes: bool = False


@dataclass
class GraphConfig:
    spatial_k: int = 6
    rna_k: int = 10
    morphology_k: int = 10
    protein_k: int = 10


@dataclass
class ModelConfig:
    widths: list[int] = field(default_factory=lambda: [512, 256, 256])
    heads: int = 4
    feature_gate_heads: int = 4
    feature_gate_width: int = 64
    scorer_width: int = 64
    decoder_width: int = 256
    protein_branch_width: int = 64
    dropout: float = 0.2
    attention_dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 300
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    align_warmup: int = 20
    lambda_rna: float = 1.0
    lambda_protein: float = 1.0
    lambda_prediction: float = 5.0
    lambda_cross: float = 0.1
    lambda_alignment: float = 0.1
    early_start: int = 30
    patience: int = 30
    min_delta: float = 0.0001
    scheduler_patience: int = 10
    min_lr: float = 0.000001
    seed: int = 0
    device: str = "auto"
    threads: int = 4
    deterministic: bool = True


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = values or {}
        known = {f.name for f in fields(cls)}
        if set(values) - known:
            raise ValueError(f"Unknown config groups: {set(values) - known}")
        result = cls(**{name: kind(**values.get(name, {})) for name, kind in
                        [("preprocess", PreprocessConfig), ("graph", GraphConfig),
                         ("model", ModelConfig), ("training", TrainConfig)]})
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream))

    def validate(self):
        p, g, m, t = self.preprocess, self.graph, self.model, self.training
        if (len(m.widths) != 3 or m.heads < 1 or
                any(w < 1 or w % m.heads for w in m.widths)):
            raise ValueError("Three encoder widths divisible by attention heads are required")
        if any(k < 1 for k in asdict(g).values()):
            raise ValueError("Graph neighbor counts must be positive")
        if not (0 <= p.max_mito_fraction <= 1 and 0 <= p.min_gene_fraction <= 1):
            raise ValueError("QC fractions must lie in [0, 1]")
        if p.min_genes < 0 or p.patch_size < 2 or p.pca_components < 1:
            raise ValueError("Invalid gene count, patch size or PCA dimension")
        if p.library_size <= 0 or p.clip <= 0:
            raise ValueError("Library size and clipping bound must be positive")
        if not (0 <= m.dropout < 1 and 0 <= m.attention_dropout < 1):
            raise ValueError("Dropout must lie in [0, 1)")
        if min(m.feature_gate_heads, m.feature_gate_width, m.scorer_width,
               m.decoder_width, m.protein_branch_width) < 1:
            raise ValueError("Model dimensions must be positive")
        if min(t.epochs, t.patience, t.scheduler_patience, t.threads) < 1:
            raise ValueError("Epochs, patience and threads must be positive")
        if min(t.learning_rate, t.min_lr, t.grad_clip) <= 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimization settings")
        if t.align_warmup < 0 or t.early_start < 0 or t.min_delta < 0:
            raise ValueError("Warmup, early-stop start and minimum improvement must be nonnegative")
        if min(t.lambda_rna, t.lambda_protein, t.lambda_prediction,
               t.lambda_cross, t.lambda_alignment) < 0:
            raise ValueError("Loss weights must be nonnegative")
