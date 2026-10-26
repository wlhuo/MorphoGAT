"""Architecture and gradient contracts; require the actual model dependencies."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from morphogat.config import ModelConfig, TrainConfig
from morphogat.losses import joint_loss
from morphogat.model import MorphoGAT, VARIANTS


@pytest.fixture
def section():
    torch.manual_seed(4)
    edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return {"x": torch.rand(4, 5), "y": torch.rand(4, 3), "f": torch.rand(4, 65),
            "mask": torch.tensor([True, False, True, False]),
            "graphs": {"spatial": edge, "rna": edge, "morphology": torch.tensor([[0, 2], [2, 0]]),
                       "protein": edge}}


@pytest.fixture
def model():
    config = ModelConfig(widths=[16, 8, 8], heads=2, feature_gate_heads=2,
                         feature_gate_width=8, decoder_width=16, protein_branch_width=8)
    return MorphoGAT(5, 3, config)


def test_inference_needs_no_proteins_and_missing_mask_is_exact(model, section):
    model.eval()
    prediction_input = {key: value for key, value in section.items() if key != "y"}
    prediction_input["graphs"] = {k: v for k, v in section["graphs"].items() if k != "protein"}
    with torch.no_grad():
        result = model.predict(prediction_input)
    missing = ~section["mask"]
    assert result["prediction"].shape == (4, 3)
    torch.testing.assert_close(result["view_weights"].sum(dim=1), torch.ones(4))
    assert torch.all(result["view_weights"][missing, 2] == 0)
    assert torch.all(result["fusion_gates"][missing] == 0)
    assert torch.equal(result["fused_latent"][missing], result["rna_latent"][missing])


def test_stop_gradient_protein_teacher(model, section):
    output = model(section)
    (output["cross_rna"].square().mean() +
     torch.nn.functional.mse_loss(output["fused_latent"], output["protein_latent"].detach())).backward()
    assert all(p.grad is None for p in model.protein_encoder.parameters())
    assert any(p.grad is not None for p in model.rna_encoder.parameters())
    assert any(p.grad is not None for p in model.rna_decoder.parameters())
    model.zero_grad(set_to_none=True)
    model(section)["protein_reconstruction"].square().mean().backward()
    assert any(p.grad is not None for p in model.protein_encoder.parameters())


def test_joint_loss_backpropagates_and_warmup(model, section):
    output = model(section)
    config = TrainConfig()
    total, terms = joint_loss(output, section, config, epoch=10)
    expected = terms["rna"] + terms["protein"] + 5 * terms["prediction"] + 0.1 * terms["cross"] + 0.05 * terms["alignment"]
    torch.testing.assert_close(total, expected)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ablation_paths(model, section, variant):
    model.variant = variant
    model.eval()
    with torch.no_grad():
        output = model.predict(section)
    assert torch.isfinite(output["prediction"]).all()
    if variant in {"no_histology", "no_morphology_graph"}:
        assert torch.all(output["view_weights"][:, 2] == 0)
    if variant in {"no_histology", "no_image_fusion"}:
        assert torch.equal(output["fused_latent"], output["rna_latent"])
    if variant == "no_adaptive_gating":
        assert torch.all(output["fusion_gates"][section["mask"]] == 1)


def test_disabled_histology_is_invariant(model, section):
    model.variant = "no_histology"
    model.eval()
    with torch.no_grad():
        first = model.predict(section)["prediction"]
        section["f"] = section["f"] * 100
        section["graphs"]["morphology"] = torch.empty((2, 0), dtype=torch.long)
        second = model.predict(section)["prediction"]
    assert torch.equal(first, second)
