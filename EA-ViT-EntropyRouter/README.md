# EA-ViT Entropy Router
Entropy-conditioned EA-ViT variant built in parallel to the original `EA-ViT` codebase.

This directory preserves stage 1 and changes only the bootstrap search plus stage 2. The router no longer receives a scalar MAC budget. Instead, it receives a flattened patch-entropy map computed from 14x14 image patches. To keep training batched, datasets are sorted by average entropy and chunked into entropy-similar batches. During inference, routing is performed one image at a time.

## Pipeline

1. Stage 1 remains unchanged and trains the elastic/nested ViT backbone.
2. The bootstrap stage still runs NSGA, but now also exports an entropy-to-subnetwork lookup table from entropy-sorted batches.
3. Stage 2 trains the router from entropy vectors first with the backbone frozen, then jointly with the ViT.

## Quick Start

```bash
# Stage 1 training (unchanged)
bash stage1.sh

# Bootstrap search + entropy lookup export
bash search_submodel.sh

# Entropy-conditioned stage 2 training
bash stage2.sh

# Single-image inference with entropy routing
bash inference.sh
```

## Notes

- The original codebase in `../EA-ViT` is not modified.
- Entropy-aware batching is implemented only for the copied variant in this directory.
- `search_submodel.sh` writes the NSGA population CSV and an entropy lookup CSV.
- `stage2.sh` consumes that lookup CSV to supervise the router and to define the MAC target used in the stage-2 constraint loss.
