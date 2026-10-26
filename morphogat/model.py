"""Shared multiview GAT, masked image fusion, and paired training heads."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

VARIANTS = ("full", "no_histology", "no_image_fusion", "no_morphology_graph", "no_adaptive_gating")
VIEWS = ("spatial", "rna", "morphology")


class FeatureGates(nn.Module):
    def __init__(self, width, hidden, heads):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(),
                          nn.Linear(hidden, width), nn.Sigmoid()) for _ in range(heads)
        ])

    def forward(self, x):
        gates = torch.stack([head(x) for head in self.heads]).mean(dim=0)
        return x * gates


class ResidualGAT(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        widths = [input_dim, *config.widths]
        self.convs = nn.ModuleList([
            GATConv(a, b // config.heads, heads=config.heads, concat=True,
                    dropout=config.attention_dropout, negative_slope=0.2,
                    add_self_loops=True, edge_dim=1, fill_value=1.0)
            for a, b in zip(widths[:-1], widths[1:])
        ])
        self.skips = nn.ModuleList([nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.norms = nn.ModuleList([nn.LayerNorm(b) for b in config.widths])
        self.feature_gates = FeatureGates(config.widths[0], config.feature_gate_width,
                                          config.feature_gate_heads)
        self.dropout = config.dropout

    def forward(self, x, edge_index):
        edge_attr = x.new_ones((edge_index.shape[1], 1))
        for i, (conv, skip, norm) in enumerate(zip(self.convs, self.skips, self.norms)):
            x = conv(x, edge_index, edge_attr) + skip(x)
            x = F.dropout(F.leaky_relu(norm(x), 0.2), p=self.dropout, training=self.training)
            if i == 0:
                x = self.feature_gates(x)
        return x


class ViewAggregation(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.view_embedding = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.view_embedding, std=0.02)
        self.scorer = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.norm = nn.LayerNorm(width)

    def forward(self, views, image_valid):
        values = torch.stack(views, dim=1)
        scores = self.scorer(values + self.view_embedding[None]).squeeze(-1)
        available = torch.ones_like(scores, dtype=torch.bool)
        available[:, 2] = image_valid
        weights = torch.softmax(scores.masked_fill(~available, -torch.inf), dim=1)
        return self.norm((weights[..., None] * values).sum(dim=1)), weights


class ImageFusion(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(65, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, width))
        self.gate = nn.Linear(width * 2, width)
        nn.init.constant_(self.gate.bias, -1.0)
        self.norm = nn.LayerNorm(width)

    def forward(self, rna, image_features, valid, fixed=False):
        image = self.project(image_features)
        gate = torch.ones_like(image) if fixed else torch.sigmoid(self.gate(torch.cat([rna, image], dim=1)))
        gate = gate * valid[:, None].to(gate.dtype)
        fused = self.norm(rna + gate * image)
        # Keep the missing-image path exactly equal, including normalization.
        return torch.where(valid[:, None], fused, rna), gate


class RNADecoder(nn.Module):
    def __init__(self, latent, genes, config):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(latent, config.decoder_width),
                                  nn.LayerNorm(config.decoder_width), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(config.decoder_width, genes))
        self.skip = nn.Linear(latent, genes, bias=False)

    def forward(self, z):
        return self.main(z) + self.skip(z)


class ProteinDecoder(nn.Module):
    def __init__(self, latent, proteins, config):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(latent, config.decoder_width),
                                    nn.LayerNorm(config.decoder_width), nn.GELU(), nn.Dropout(config.dropout))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(config.decoder_width, config.protein_branch_width),
                          nn.GELU(), nn.Dropout(config.dropout),
                          nn.Linear(config.protein_branch_width, 1)) for _ in range(proteins)
        ])

    def forward(self, z):
        shared = self.shared(z)
        return torch.cat([branch(shared) for branch in self.branches], dim=1)


class MorphoGAT(nn.Module):
    def __init__(self, genes, proteins, config, variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        latent = config.widths[-1]
        self.rna_encoder = ResidualGAT(genes, config)
        self.protein_encoder = ResidualGAT(proteins, config)
        self.aggregation = ViewAggregation(latent, config.scorer_width)
        self.fusion = ImageFusion(latent, config.decoder_width, config.dropout)
        self.rna_decoder = RNADecoder(latent, genes, config)
        self.protein_decoder = ProteinDecoder(latent, proteins, config)

    def encode_rna(self, section):
        graphs = section["graphs"]
        views = [self.rna_encoder(section["x"], graphs[v]) for v in VIEWS[:2]]
        morphology = self.variant not in {"no_histology", "no_morphology_graph"}
        views.append(self.rna_encoder(section["x"], graphs["morphology"]) if morphology
                     else torch.zeros_like(views[0]))
        image_valid = section["mask"] if morphology else torch.zeros_like(section["mask"])
        rna, weights = self.aggregation(views, image_valid)
        if self.variant in {"no_histology", "no_image_fusion"}:
            fused, gates = rna, torch.zeros_like(rna)
        else:
            fused, gates = self.fusion(rna, section["f"], section["mask"],
                                       fixed=self.variant == "no_adaptive_gating")
        return rna, fused, weights, gates

    def predict(self, section):
        rna, fused, weights, gates = self.encode_rna(section)
        return {"prediction": self.protein_decoder(fused), "rna_latent": rna,
                "fused_latent": fused, "view_weights": weights, "fusion_gates": gates}

    def forward(self, section):
        output = self.predict(section)
        protein = self.protein_encoder(section["y"], section["graphs"]["protein"])
        output.update({"rna_reconstruction": self.rna_decoder(output["rna_latent"]),
                       "protein_reconstruction": self.protein_decoder(protein),
                       "cross_rna": self.rna_decoder(protein.detach()),
                       "protein_latent": protein})
        return output
