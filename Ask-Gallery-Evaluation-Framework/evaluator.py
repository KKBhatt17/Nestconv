import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def extract_image_name(path: str) -> str:
    normalized = path.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def longest_common_subsequence_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0

    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for index_b, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_token_overlap_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    overlap = sum((pred_counter & gt_counter).values())

    precision = safe_divide(overlap, len(pred_tokens))
    recall = safe_divide(overlap, len(gt_tokens))
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision + recall else 0.0

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute_rouge_l(prediction: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    lcs = longest_common_subsequence_length(pred_tokens, gt_tokens)

    precision = safe_divide(lcs, len(pred_tokens))
    recall = safe_divide(lcs, len(gt_tokens))
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision + recall else 0.0

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


def compute_substring_inclusion(prediction: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0
    return 1.0 if ground_truth in prediction else 0.0


@dataclass
class MetricSwitches:
    any_recall: bool = True
    all_recall: bool = True
    mrr: bool = True
    token_metrics: bool = True
    rouge_l: bool = True
    substring_inclusion: bool = True
    llm_judge: bool = False


@dataclass
class LLMConfig:
    run_extraction: bool = False
    provider: str = "ollama"
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    temperature: float = 0.0


@dataclass
class EvaluationConfig:
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    metrics: MetricSwitches = field(default_factory=MetricSwitches)
    llm: LLMConfig = field(default_factory=LLMConfig)
    answer_gt_aggregation: str = "best"


@dataclass
class QueryGroundTruth:
    query: str
    image_answers: List[Tuple[str, str]]

    @property
    def image_names(self) -> List[str]:
        return [extract_image_name(image_path) for image_path, _ in self.image_answers]

    @property
    def answers(self) -> List[str]:
        return [answer for _, answer in self.image_answers]


@dataclass
class QueryPrediction:
    retrieved_images: List[str]
    answer: str


class EvaluatorError(Exception):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        provider = self.config.provider.lower()
        if provider == "ollama":
            return self._generate_with_ollama(prompt)
        if provider in {"openai", "openai_compatible"}:
            return self._generate_with_openai_compatible(prompt)
        raise EvaluatorError(f"Unsupported LLM provider: {self.config.provider}")

    def _generate_with_ollama(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        return self._post_json(url, payload, headers={}).get("response", "").strip()

    def _generate_with_openai_compatible(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise evaluator. Follow the user instructions exactly.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = self._post_json(url, payload, headers=headers)
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise EvaluatorError("Unexpected OpenAI-compatible response shape.") from exc

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise EvaluatorError(f"LLM request failed for {url}: {exc}") from exc


def extract_key_information(text: str, llm_client: Optional[LLMClient], enabled: bool) -> str:
    if not enabled or llm_client is None:
        return text

    prompt = (
        "Extract only the minimal factual answer from the text below.\n"
        "Return only the extracted answer, with no explanation.\n"
        "If the text is already minimal, return it unchanged.\n\n"
        f"Text: {text}"
    )
    response = llm_client.generate(prompt).strip()
    return response or text


def llm_judge_contains_fact(
    ground_truth: str,
    prediction: str,
    llm_client: Optional[LLMClient],
) -> float:
    if llm_client is None:
        raise EvaluatorError("LLM judge metric was requested, but no LLM client is configured.")

    prompt = (
        "You are an evaluator.\n"
        f"Ground Truth: {ground_truth}\n"
        f"Model Prediction: {prediction}\n"
        "Does the prediction contain the exact factual information required by the ground truth, "
        "even if it includes extra conversational text?\n"
        "Answer strictly YES or NO."
    )
    response = llm_client.generate(prompt).strip().upper()
    if response.startswith("YES"):
        return 1.0
    if response.startswith("NO"):
        return 0.0
    raise EvaluatorError(f"Unexpected LLM judge response: {response}")


def parse_ground_truth(data: Dict[str, Any]) -> Dict[str, QueryGroundTruth]:
    parsed: Dict[str, QueryGroundTruth] = {}
    for query, items in data.items():
        if not isinstance(items, list):
            raise EvaluatorError(f"Ground truth for query '{query}' must be a list.")

        image_answers: List[Tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                raise EvaluatorError(
                    f"Each ground-truth entry for query '{query}' must be a single-key object."
                )
            image_path, answer = next(iter(item.items()))
            image_answers.append((str(image_path), str(answer)))

        parsed[query] = QueryGroundTruth(query=query, image_answers=image_answers)
    return parsed


def parse_prediction_record(record: Any, query: str) -> QueryPrediction:
    if isinstance(record, dict):
        retrieved_images = (
            record.get("retrieved_images")
            or record.get("retrieval")
            or record.get("images")
            or record.get("predicted_images")
            or []
        )
        answer = record.get("answer") or record.get("prediction") or record.get("predicted_answer") or ""
    else:
        raise EvaluatorError(
            f"Prediction for query '{query}' must be an object containing retrieved images and answer."
        )

    if not isinstance(retrieved_images, list):
        raise EvaluatorError(f"Retrieved images for query '{query}' must be a list.")

    image_names = [extract_image_name(str(image_name)) for image_name in retrieved_images]
    return QueryPrediction(retrieved_images=image_names, answer=str(answer))


def parse_predictions(data: Any) -> Dict[str, QueryPrediction]:
    if isinstance(data, dict):
        return {query: parse_prediction_record(record, query) for query, record in data.items()}

    if isinstance(data, list):
        parsed: Dict[str, QueryPrediction] = {}
        for item in data:
            if not isinstance(item, dict) or "query" not in item:
                raise EvaluatorError("Prediction lists must contain objects with a 'query' field.")
            query = str(item["query"])
            parsed[query] = parse_prediction_record(item, query)
        return parsed

    raise EvaluatorError("Predictions must be either a query-keyed object or a list of query records.")


def reciprocal_rank(retrieved_images: Sequence[str], relevant_images: Sequence[str]) -> float:
    relevant = set(relevant_images)
    for index, image_name in enumerate(retrieved_images, start=1):
        if image_name in relevant:
            return 1.0 / index
    return 0.0


def aggregate_scores(score_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not score_rows:
        return {}

    keys = sorted({key for row in score_rows for key in row.keys()})
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in score_rows if key in row]
        aggregated[key] = sum(values) / len(values) if values else 0.0
    return aggregated


def build_answer_candidates(
    gt_answers: Sequence[str],
    aggregation_mode: str,
) -> List[Tuple[str, str]]:
    candidates = [(f"image_{index + 1}", answer) for index, answer in enumerate(gt_answers)]

    if aggregation_mode == "best":
        return candidates
    if aggregation_mode == "concat":
        concatenated = " ".join(answer for answer in gt_answers if answer).strip()
        return [("concat", concatenated)] if concatenated else candidates
    if aggregation_mode == "best_and_concat":
        concatenated = " ".join(answer for answer in gt_answers if answer).strip()
        if concatenated:
            candidates.append(("concat", concatenated))
        return candidates

    raise EvaluatorError(
        "answer_gt_aggregation must be one of: 'best', 'concat', 'best_and_concat'."
    )


def score_answer_against_candidates(
    prediction_answer: str,
    gt_answers: Sequence[str],
    config: EvaluationConfig,
    llm_client: Optional[LLMClient],
) -> Dict[str, Any]:
    processed_prediction = extract_key_information(
        prediction_answer,
        llm_client=llm_client,
        enabled=config.llm.run_extraction,
    )
    normalized_prediction = normalize_text(processed_prediction)
    candidates = build_answer_candidates(gt_answers, config.answer_gt_aggregation)

    best_candidate_label = None
    best_candidate_text = None
    best_score_sum = -math.inf
    best_metrics: Dict[str, float] = {}

    for candidate_label, candidate_text in candidates:
        processed_candidate = extract_key_information(
            candidate_text,
            llm_client=llm_client,
            enabled=config.llm.run_extraction,
        )
        normalized_candidate = normalize_text(processed_candidate)
        metrics: Dict[str, float] = {}

        if config.metrics.token_metrics:
            metrics.update(compute_token_overlap_metrics(normalized_prediction, normalized_candidate))
        if config.metrics.rouge_l:
            metrics.update(compute_rouge_l(normalized_prediction, normalized_candidate))
        if config.metrics.substring_inclusion:
            metrics["substring_inclusion"] = compute_substring_inclusion(
                normalized_prediction,
                normalized_candidate,
            )
        if config.metrics.llm_judge:
            metrics["llm_judge"] = llm_judge_contains_fact(
                ground_truth=processed_candidate,
                prediction=processed_prediction,
                llm_client=llm_client,
            )

        score_sum = sum(metrics.values())
        if score_sum > best_score_sum:
            best_score_sum = score_sum
            best_candidate_label = candidate_label
            best_candidate_text = candidate_text
            best_metrics = metrics

    return {
        "selected_ground_truth_answer": best_candidate_text,
        "selected_ground_truth_source": best_candidate_label,
        "processed_prediction_answer": processed_prediction,
        "normalized_prediction_answer": normalized_prediction,
        "metrics": best_metrics,
    }


def evaluate(
    ground_truth: Dict[str, QueryGroundTruth],
    predictions: Dict[str, QueryPrediction],
    config: EvaluationConfig,
) -> Dict[str, Any]:
    llm_client = None
    if config.llm.run_extraction or config.metrics.llm_judge:
        llm_client = LLMClient(config.llm)

    per_query_results: Dict[str, Any] = {}
    retrieval_rows_by_k: Dict[int, List[Dict[str, float]]] = {k: [] for k in config.k_values}
    mrr_rows: List[Dict[str, float]] = []
    answer_rows: List[Dict[str, float]] = []
    missing_prediction_queries: List[str] = []
    extra_prediction_queries = sorted(set(predictions.keys()) - set(ground_truth.keys()))

    for query, gt_item in ground_truth.items():
        prediction = predictions.get(query)
        if prediction is None:
            missing_prediction_queries.append(query)
            prediction = QueryPrediction(retrieved_images=[], answer="")

        query_result: Dict[str, Any] = {
            "ground_truth_images": gt_item.image_names,
            "predicted_images": prediction.retrieved_images,
            "ground_truth_answers": gt_item.answers,
        }

        for k in config.k_values:
            top_k = prediction.retrieved_images[:k]
            relevant = set(gt_item.image_names)
            retrieval_metrics: Dict[str, float] = {}

            if config.metrics.any_recall:
                retrieval_metrics["any_recall"] = 1.0 if relevant.intersection(top_k) else 0.0
            if config.metrics.all_recall:
                retrieval_metrics["all_recall"] = 1.0 if relevant.issubset(set(top_k)) else 0.0

            query_result.setdefault("retrieval", {})[str(k)] = retrieval_metrics
            retrieval_rows_by_k[k].append(retrieval_metrics)

        if config.metrics.mrr:
            mrr_metrics = {"mrr": reciprocal_rank(prediction.retrieved_images, gt_item.image_names)}
            query_result["mrr"] = mrr_metrics["mrr"]
            mrr_rows.append(mrr_metrics)

        answer_result = score_answer_against_candidates(
            prediction_answer=prediction.answer,
            gt_answers=gt_item.answers,
            config=config,
            llm_client=llm_client,
        )
        query_result["predicted_answer"] = prediction.answer
        query_result["answer"] = answer_result
        answer_rows.append(answer_result["metrics"])
        per_query_results[query] = query_result

    aggregated_retrieval_by_k = {
        f"k={k}": aggregate_scores(rows) for k, rows in retrieval_rows_by_k.items()
    }
    aggregated_retrieval: Dict[str, Any] = {"by_k": aggregated_retrieval_by_k}
    if config.metrics.mrr:
        aggregated_retrieval["mrr"] = aggregate_scores(mrr_rows).get("mrr", 0.0)
    aggregated_answers = aggregate_scores(answer_rows)

    return {
        "config": {
            "k_values": config.k_values,
            "answer_gt_aggregation": config.answer_gt_aggregation,
            "metrics": vars(config.metrics),
            "llm": {
                "run_extraction": config.llm.run_extraction,
                "provider": config.llm.provider,
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "timeout_seconds": config.llm.timeout_seconds,
            },
        },
        "summary": {
            "num_queries": len(ground_truth),
            "num_missing_predictions": len(missing_prediction_queries),
            "num_extra_predictions": len(extra_prediction_queries),
            "missing_prediction_queries": missing_prediction_queries,
            "extra_prediction_queries": extra_prediction_queries,
            "retrieval": aggregated_retrieval,
            "answering": aggregated_answers,
        },
        "per_query": per_query_results,
    }


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_config(config_data: Optional[Dict[str, Any]]) -> EvaluationConfig:
    if not config_data:
        return EvaluationConfig()

    metrics_data = config_data.get("metrics", {})
    llm_data = config_data.get("llm", {})

    metrics = MetricSwitches(
        any_recall=metrics_data.get("any_recall", True),
        all_recall=metrics_data.get("all_recall", True),
        mrr=metrics_data.get("mrr", True),
        token_metrics=metrics_data.get("token_metrics", True),
        rouge_l=metrics_data.get("rouge_l", True),
        substring_inclusion=metrics_data.get("substring_inclusion", True),
        llm_judge=metrics_data.get("llm_judge", False),
    )
    llm = LLMConfig(
        run_extraction=llm_data.get("run_extraction", False),
        provider=llm_data.get("provider", "ollama"),
        model=llm_data.get("model", "qwen3:4b"),
        base_url=llm_data.get("base_url", "http://localhost:11434"),
        api_key=llm_data.get("api_key") or os.getenv("OPENAI_API_KEY"),
        timeout_seconds=llm_data.get("timeout_seconds", 30),
        temperature=llm_data.get("temperature", 0.0),
    )

    return EvaluationConfig(
        k_values=config_data.get("k_values", [1, 3, 5]),
        metrics=metrics,
        llm=llm,
        answer_gt_aggregation=config_data.get("answer_gt_aggregation", "best"),
    )


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Retrieval+VQA predictions against a GT JSON."
    )
    parser.add_argument("--ground-truth", required=True, help="Path to the ground-truth JSON.")
    parser.add_argument("--predictions", required=True, help="Path to the predictions JSON.")
    parser.add_argument(
        "--config",
        required=False,
        help="Optional config JSON controlling metric switches, k values, and LLM settings.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output JSON path. If omitted, results are printed to stdout.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        ground_truth_data = load_json(args.ground_truth)
        prediction_data = load_json(args.predictions)
        config_data = load_json(args.config) if args.config else None

        ground_truth = parse_ground_truth(ground_truth_data)
        predictions = parse_predictions(prediction_data)
        config = parse_config(config_data)

        results = evaluate(ground_truth, predictions, config)

        if args.output:
            save_json(args.output, results)
        else:
            json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
        return 0
    except EvaluatorError as exc:
        print(f"Evaluator error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
