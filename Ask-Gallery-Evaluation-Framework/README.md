# Retrieval + VQA Evaluation Framework

This repo contains a lightweight Python evaluator for a two-stage Retrieval + VQA pipeline:

1. Retrieve top-k images for a text query.
2. Generate one final answer string from the retrieved images.

The evaluator supports:

- Retrieval metrics: `Any-Recall@k`, `All-Recall@k`, `MRR`
- Answer metrics: token precision/recall/F1, `ROUGE-L`, substring inclusion, optional `LLM-as-a-judge`
- Optional LLM-based key-information extraction before scoring
- Per-metric switches so you can enable only the metrics you want

## Expected Ground Truth Format

```json
{
  "what is my pnr for the flight to delhi on 2nd oct?": [
    {
      "C:/data/boarding_pass.png": "WYX567"
    }
  ],
  "what is my most recent journey plan?": [
    {
      "C:/data/trip_1.png": "Delhi to Mumbai on 2 Oct"
    },
    {
      "C:/data/trip_2.png": "Mumbai to Bangalore on 6 Oct"
    }
  ]
}
```

The evaluator automatically converts each GT image path to its image name, such as `boarding_pass.png`.

## Expected Prediction Format

Recommended query-keyed JSON:

```json
{
  "what is my pnr for the flight to delhi on 2nd oct?": {
    "retrieved_images": ["boarding_pass.png", "itinerary.png"],
    "answer": "The PNR is WYX567."
  },
  "what is my most recent journey plan?": {
    "retrieved_images": ["trip_2.png", "trip_1.png"],
    "answer": "Mumbai to Bangalore on 6 Oct"
  }
}
```

It also accepts a list of records:

```json
[
  {
    "query": "what is my pnr for the flight to delhi on 2nd oct?",
    "retrieved_images": ["boarding_pass.png", "itinerary.png"],
    "answer": "The PNR is WYX567."
  }
]
```

## Config Format

All metrics are individually switchable.

```json
{
  "k_values": [1, 3, 5],
  "answer_gt_aggregation": "best",
  "metrics": {
    "any_recall": true,
    "all_recall": true,
    "mrr": true,
    "token_metrics": true,
    "rouge_l": true,
    "substring_inclusion": true,
    "llm_judge": false
  },
  "llm": {
    "run_extraction": false,
    "provider": "ollama",
    "model": "qwen3:4b",
    "base_url": "http://localhost:11434",
    "timeout_seconds": 30,
    "temperature": 0.0
  }
}
```

`answer_gt_aggregation` options:

- `best`: compare the prediction against each GT answer for the query and keep the best-scoring one
- `concat`: concatenate all GT answers into one answer before scoring
- `best_and_concat`: allow both individual GT answers and the concatenated answer, then keep the best

`best` is the default because your system returns a single answer that may come from any one relevant image.

## Answer Normalization

Before answer metrics are computed, the evaluator applies:

- Lowercasing
- Punctuation stripping
- Whitespace normalization

If `llm.run_extraction` is enabled, it first tries to extract the minimal factual answer, for example `ABC123` from `The PNR is ABC123`.

## LLM Options

Two LLM-assisted features are available:

1. Key-information extraction before scoring
2. LLM-as-a-judge for verbose answer handling

Supported providers:

- `ollama`
- `openai_compatible`

For `ollama`, serve a local model such as `qwen3`, `gemma3`, or `llama3` and point `base_url` to the Ollama server, usually `http://localhost:11434`.

For `openai_compatible`, set `base_url` to your chat-completions endpoint and provide `api_key` in config or `OPENAI_API_KEY` in the environment.

## Run

```powershell
python evaluator.py --ground-truth ground_truth.json --predictions predictions.json --config config.json --output results.json
```

If `--output` is omitted, the result JSON is printed to stdout.

## Output

The output JSON contains:

- `summary`: aggregate metrics across all queries
- `per_query`: query-level retrieval and answer details
- `config`: the resolved config used for evaluation

## Notes

- `Any-Recall@k` and `All-Recall@k` are reported under `summary.retrieval.by_k`.
- `MRR` is reported once over the full retrieved ranking under `summary.retrieval.mrr`.
- Missing predictions are scored as empty retrieval and empty answer.
- Extra predictions that do not exist in GT are listed in the summary.

## Tests

```powershell
python -m unittest discover -s tests -v
```
