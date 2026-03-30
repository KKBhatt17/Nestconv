# EA-VIT codebase

Your “3-stage” reading is close to the script flow, but the repo itself treats it as a 2-stage method with an NSGA search/bootstrap step in between. The README says: stage 1 learns an elastic backbone, then stage 2 learns a router and jointly refines it with the backbone ([README.md#L37](C:/Users/krish/Documents/Nestconv/EA-ViT/README.md#L37), [README.md#L9](C:/Users/krish/Documents/Nestconv/EA-ViT/README.md#L9)).

**Big Picture**
The pipeline is:

1. Start from a “rearranged by importance” pretrained ViT checkpoint.
2. Train an elastic ViT backbone that can expose many subnetworks from one set of weights.
3. Run an NSGA-style search over explicit subnetwork masks to build a budget/accuracy lookup table.
4. Train a router that maps a scalar compute constraint to masks, first with router-only warmup, then with joint training.
5. At inference, give a compute budget and let the router choose the active subnetwork.

That checkpoint rearrangement matters because the elastic model always keeps prefix slices of channels/heads/MLP units, not arbitrary subsets. You can see that directly in the slicing logic in [model_stage1.py#L67](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage1.py#L67), [model_stage1.py#L125](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage1.py#L125), and [model_stage1.py#L198](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage1.py#L198).

**Stage 1: Elastic Backbone Adaptation**
The backbone is a `timm` ViT with custom elastic replacements for LayerNorm, attention, MLP, and head in [model_stage1.py#L164](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage1.py#L164). `configure_subnetwork(...)` sets 4 elastic axes: embedding dimension, attention head count, MLP ratio, and which transformer blocks are active ([model_stage1.py#L204](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage1.py#L204)).

The important implementation idea is:
- Embedding width is truncated by taking the first `sub_dim` channels.
- Attention uses only the first `num_heads * head_dim` QKV rows and the matching projection columns.
- MLP uses only the first `sub_dim * ratio` hidden units.
- Depth is controlled by iterating only over `depth_list`.

Training is in [train_stage1.py](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py). Each batch samples a random subnetwork from a curriculum-expanded search space: `mlp_ratio_list`, `mha_head_list`, `sub_dim = 64 * heads`, plus a reduced block list ([train_stage1.py#L19](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py#L19), [train_stage1.py#L91](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py#L91)). Validation probes 5 fixed presets labeled `l/m/s/ss/sss` ([train_stage1.py#L132](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py#L132), [eval_flag.py#L5](C:/Users/krish/Documents/Nestconv/EA-ViT/utils/eval_flag.py#L5)).

Conceptually, this stage teaches one backbone to survive many width/head/MLP/depth settings.

**Search Step: NSGA Guide For The Router**
`search_submodel.py` is the bridge between stage 1 and stage 2. It loads the stage-1 weights into the stage-2 model with `strict=False`, so the new router parameters can stay randomly initialized while backbone weights transfer ([search_submodel.py#L22](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L22), [search_submodel.py#L94](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L94)).

Its genome is 276 binary values:
- 12 embed-dim chunks
- 12 attention-depth gates
- 12 MLP-depth gates
- 12 × 12 attention-head chunks
- 12 × 8 MLP chunks

That layout is defined around [search_submodel.py#L34](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L34). But one subtlety: the code does not use those bits as arbitrary masks. It sums each group and converts it into a prefix mask like `[1,1,1,0,0,...]` ([search_submodel.py#L100](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L100), [search_submodel.py#L109](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L109)). So the search space is really “how many units to keep,” not “which exact units to keep.”

Fitness is `(MACs, accuracy)`, using the stage-2 model’s internal cost estimate and classification accuracy from a forward pass ([search_submodel.py#L94](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L94)). Each generation gets appended to CSV, and stage 2 later reads one chosen generation, usually `gen_id=300` ([search_submodel.py#L160](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L160), [train_stage2.py#L125](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L125)).

**Stage 2: Router + Joint Dynamic Model**
The routerized model is in [model_stage2.py#L292](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L292). It adds:
- One embed router
- One attention-depth router
- One MLP-depth router
- One attention router per block
- One MLP router per block

All routers take a scalar constraint and produce mask logits. The masks are sampled with straight-through Gumbel-sigmoid in [_gumbel_sigmoid](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L14). Mask construction is centralized in [model_stage2.py#L440](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L440).

A few concrete details:
- Embed mask has 12 chunks of 64 dims; the first 5 are always forced on, so router-controlled width is effectively 320 to 768 dims ([model_stage2.py#L445](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L445)).
- Attention head mask also forces the first 5 heads on, then routes the remaining 7 ([model_stage2.py#L158](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L158)).
- MLP mask has 8 chunks, which corresponds to expansion-ratio steps of `0.5`, because 3072 hidden dims are grouped into 8 chunks of 384 ([model_stage2.py#L203](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L203), [model_stage2.py#L403](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L403)).
- Depth is split into separate gates for attention and MLP residual branches ([model_stage2.py#L327](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L327), [model_stage2.py#L281](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L281)).

During training, the model mostly keeps full tensors and multiplies masks into them. During eval, it physically slices dimensions/weights for actual smaller execution ([model_stage2.py#L346](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L346), [model_stage2.py#L355](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L355)). So the router is trained with differentiable masked computation, but inference uses true subnetwork extraction.

The stage-2 training script is [train_stage2.py](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py). Your statement “stage 3 trains the router keeping the model frozen” is only partially true:
- First, it freezes everything except parameters whose names contain `router` ([train_stage2.py#L90](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L90)).
- Then it does a short router-only warmup loop ([train_stage2.py#L129](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L129)).
- After that, it unfreezes the whole model and continues joint training ([train_stage2.py#L182](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L182)).

The stage-2 loss is:
- Cross-entropy on task labels
- Constraint loss: predicted normalized MACs should match the requested budget
- Label-mask loss: router masks should match the nearest NSGA-searched mask for that budget

See [train_stage2.py#L147](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L147) through [train_stage2.py#L157](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L157). Later, the NSGA guidance is decayed over epochs with `* (1 - epoch / args.epochs)` ([train_stage2.py#L240](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L240)). So NSGA acts like a teacher/initializer, not a permanent hard rule.

**Inference And Supporting Files**
Datasets are selected in [image_datasets.py](C:/Users/krish/Documents/Nestconv/EA-ViT/dataloader/image_datasets.py), with torchvision-style downloads and ImageNet-Inception normalization. `inference.py` loads the stage-2 checkpoint, sets a user constraint, profiles MACs with DeepSpeed, and evaluates accuracy ([inference.py#L77](C:/Users/krish/Documents/Nestconv/EA-ViT/inference.py#L77), [inference.py#L104](C:/Users/krish/Documents/Nestconv/EA-ViT/inference.py#L104)). `entropy.py` looks like a standalone utility and is not part of the training path.

**Important “As-Committed” Caveats**
This repo has a few mismatches between intended design and literal runtime behavior:

- `config.py` defines `max_lr`, but both the LR scheduler and wandb helper use `args.lr`, which is never added to the parser ([config.py#L53](C:/Users/krish/Documents/Nestconv/EA-ViT/config.py#L53), [lr_sched.py#L3](C:/Users/krish/Documents/Nestconv/EA-ViT/utils/lr_sched.py#L3), [set_wandb.py#L3](C:/Users/krish/Documents/Nestconv/EA-ViT/utils/set_wandb.py#L3)).
- `train_stage1.py` references `args.stage_epochs`, but only `curriculum_epochs` exists ([train_stage1.py#L76](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py#L76), [config.py#L37](C:/Users/krish/Documents/Nestconv/EA-ViT/config.py#L37)).
- `search_submodel.py` calls `model.configure_latency(...)`, but the model only defines `configure_constraint(...)` ([search_submodel.py#L97](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L97), [model_stage2.py#L482](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L482)).
- `inference.sh` passes `--constrant`, but the parser expects `--constraint` ([inference.sh#L15](C:/Users/krish/Documents/Nestconv/EA-ViT/inference.sh#L15), [config.py#L45](C:/Users/krish/Documents/Nestconv/EA-ViT/config.py#L45)).
- The training loops in stage 1, stage 2 warmup, and stage 2 main training all `break` after the first batch, so as written they only process one batch per epoch/phase ([train_stage1.py#L123](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage1.py#L123), [train_stage2.py#L177](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L177), [train_stage2.py#L272](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L272)).
- The NSGA “teacher” loader reads all stored individuals for a generation, not just Pareto-flagged ones, even though the CSV stores `IsPareto` ([train_stage2.py#L22](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L22), [search_submodel.py#L168](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L168)).

So the intended design is strong and fairly clear, but this exact checked-in code looks like a partially debug-edited or inconsistent release.


**What NSGA Is Doing**

This search step is the bridge between the elastic backbone and the router. The goal is not to train the router yet. Instead, it builds a library of good subnetwork structures across the accuracy/MACs tradeoff, so stage 2 can imitate those structures for different budgets.

The script is [search_submodel.py](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py). It loads the stage-1-trained backbone into the stage-2 model so it can evaluate many candidate subnetworks on the downstream task ([search_submodel.py#L22](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L22)). The stage-2 model is used here because it already knows how to accept explicit masks and compute a normalized MACs estimate ([model_stage2.py#L377](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L377), [model_stage2.py#L387](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L387)).

**How One Candidate Architecture Is Encoded**

Each NSGA individual is a binary vector of length 276, defined at [search_submodel.py#L34](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L34). It represents one subnetwork choice:

- 12 bits for embedding width chunks
- 12 bits for attention branch depth gates
- 12 bits for MLP branch depth gates
- 12 × 12 bits for per-layer attention-head chunks
- 12 × 8 bits for per-layer MLP chunks

That is:
- `vector[:12]` controls embedding width
- `vector[12:24]` controls which blocks keep attention residuals
- `vector[24:36]` controls which blocks keep MLP residuals
- `vector[36:180]` controls attention capacity layer by layer
- `vector[180:276]` controls MLP capacity layer by layer

But there is an important nuance: the script does not use those bits as arbitrary sparse masks. It sums each region and then turns that sum into a prefix mask. You can see this in [search_submodel.py#L100](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L100) through [search_submodel.py#L117](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L117).

So if the 12 embedding bits contain 7 ones scattered anywhere, the actual embed mask becomes:

```text
[1,1,1,1,1,1,1,0,0,0,0,0]
```

not the original scattered pattern.

The same thing happens for attention heads and MLP chunks per layer. This means the real searched architecture is “how many leading chunks to keep,” not “which exact chunks to keep.” That matches the stage-1 nested design, where important channels/heads are assumed to have been rearranged to the front.

**How A Candidate Gets Evaluated**

The fitness function is `evaluate(vector)` at [search_submodel.py#L94](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L94).

For each candidate:

- It converts the genome into `embed_mask`, `depth_attn_mask`, `depth_mlp_mask`, `mask_attn`, and `mask_mlp`.
- It injects those masks into the model using `model.set_mask(...)` ([search_submodel.py#L117](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L117), [model_stage2.py#L500](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L500)).
- It runs a forward pass on one training batch and gets both predictions and the model’s internal MACs estimate ([search_submodel.py#L122](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L122), [model_stage2.py#L383](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L383)).
- It returns a 2-objective fitness:
  - minimize MACs
  - maximize accuracy

That multi-objective definition is created here: [search_submodel.py#L40](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L40).

One subtle point: the “accuracy” used during search is only measured on the first batch because the loop breaks immediately after one batch ([search_submodel.py#L122](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L122)). So this is a very cheap and noisy proxy, not a full validation-set estimate.

**How The Evolution Works**

The script uses DEAP’s NSGA-style machinery. The main ingredients are:

- population size = 5 ([search_submodel.py#L38](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L38))
- generations = 301 ([search_submodel.py#L35](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L35))
- two-point crossover ([search_submodel.py#L277](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L277))
- bit-flip mutation ([search_submodel.py#L278](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L278))

The interesting part is that they do not use plain NSGA-II survivor selection. They use `select_by_partition_incremental(...)` at [search_submodel.py#L199](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L199).

That custom selector does a few things:

- Merges parents and offspring.
- Sorts by nondominated fronts.
- Computes a MACs-based crowding score with `assign_macs_global_crowding(...)` so solutions are spread across the compute axis ([search_submodel.py#L176](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L176)).
- Splits the MACs range into bins using `np.linspace(0.0, 1.0, 21)` ([search_submodel.py#L314](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L314)).
- Tries to keep diversity inside each MACs region using quotas and a minimum MACs difference ([search_submodel.py#L313](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L313), [search_submodel.py#L318](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L318)).

In plain language: they are trying to avoid ending up with many nearly identical solutions clustered at the same budget. They want the CSV to contain usable structures across the whole budget spectrum.

**What Gets Saved**

Every generation is appended to a CSV by `save_population(...)` at [search_submodel.py#L160](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L160). Each row stores:

- generation index
- MACs
- accuracy
- full binary encoding
- whether that individual is on the current Pareto front

That CSV becomes the teacher data for stage 2.

**How Stage 2 Uses It**

Stage 2 reads the CSV with `load_pareto_data(...)` in [train_stage2.py#L22](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L22). Then for each sampled constraint, `get_preset_mask_nsga(...)` finds the stored encoding whose MACs is closest to that constraint and converts it back into masks ([train_stage2.py#L35](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L35)).

Those masks become supervision for the router through `label_mask_loss` in [train_stage2.py#L151](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L151) and [train_stage2.py#L238](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L238).

So the router is learning:

- “for budget 0.2, produce masks similar to the searched low-cost submodel”
- “for budget 0.8, produce masks similar to the searched higher-capacity submodel”

That is why I called the NSGA output a “guide” or “teacher” for the router.

**Two Important Caveats In This Repo**

A couple of details matter if you are reading this code literally:

- `search_submodel.py` calls `model.configure_latency(...)` at [search_submodel.py#L97](C:/Users/krish/Documents/Nestconv/EA-ViT/search_submodel.py#L97), but the model only defines `configure_constraint(...)` in [model_stage2.py#L482](C:/Users/krish/Documents/Nestconv/EA-ViT/models/model_stage2.py#L482). That looks like a naming bug.
- Stage 2’s CSV loader does not filter to only `IsPareto == 1`; it reads all individuals in a generation ([train_stage2.py#L22](C:/Users/krish/Documents/Nestconv/EA-ViT/train_stage2.py#L22)). So in the current code, the “teacher” is really “nearest searched sample,” not strictly “nearest Pareto-optimal sample.”

The clean conceptual summary is: NSGA is used to discover good discrete subnetworks across the budget frontier, write them to disk, and turn them into pseudo-labels so the router can learn a direct mapping from scalar budget to elastic architecture.

If you want, I can next walk through one concrete example genome end-to-end and show exactly how a 276-bit vector becomes actual masks inside the transformer.
