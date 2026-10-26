"""Convert paired 10x feature-barcode H5 counts to a MorphoGAT NPZ section."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from morphogat.data import unique_names
from morphogat.images import read_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="filtered_feature_bc_matrix.h5, containing RNA and ADT")
    parser.add_argument("--positions", required=True, help="Visium tissue_positions CSV")
    parser.add_argument("--output", required=True, help="Output .npz")
    parser.add_argument("--rna-only", action="store_true")
    args = parser.parse_args()
    import scanpy as sc
    data = sc.read_10x_h5(args.matrix, gex_only=False)
    rna = data[:, data.var["feature_types"] == "Gene Expression"]
    protein = data[:, data.var["feature_types"] == "Antibody Capture"]
    ids = unique_names(data.obs_names, "barcodes")
    genes = unique_names(rna.var_names, "gene symbols")
    centers = read_positions(args.positions).reindex(ids).to_numpy(dtype=np.float32)
    if not np.isfinite(centers).all():
        raise ValueError("Each barcode requires finite full-resolution pixel coordinates")
    if rna.n_vars == 0:
        raise ValueError("The H5 matrix has no Gene Expression feature type")
    values = {"counts": rna.X.toarray().astype(np.float32), "genes": genes,
              "spot_ids": ids, "spatial": centers}
    if not args.rna_only:
        if protein.n_vars == 0:
            raise ValueError("No Antibody Capture features; use --rna-only for external data")
        values.update(protein=protein.X.toarray().astype(np.float32),
                      proteins=unique_names(protein.var_names, "protein names"))
    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        raise ValueError("The output filename must end with .npz")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    print(f"Saved {len(ids)} spots and {len(genes)} genes to {output}")
    print("Add image, scalefactors, scale_key and spatial_is_pixel: true to the section manifest.")


if __name__ == "__main__":
    main()
