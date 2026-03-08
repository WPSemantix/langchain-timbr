import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_timbr import run_benchmark

logger = logging.getLogger(__name__)


def _resolve_mechanism_flags(mechanism: str) -> Tuple[bool, bool]:
    normalized = (mechanism or "full").strip().lower()
    if normalized == "deterministic":
        return True, False
    if normalized == "llm_judge":
        return False, True
    if normalized == "full":
        return True, True
    raise ValueError("Invalid mechanism. Use one of: full, deterministic, llm_judge.")


def _resolve_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    return os.environ.get("TIMBR_BENCHMARK_TOKEN") or os.environ.get("TIMBR_TOKEN")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Timbr benchmark from CLI.")
    parser.add_argument("--benchmark-name", required=True, help="Benchmark name from SYS_AGENTS_BENCHMARKS")
    parser.add_argument("--url", required=True, help="Timbr API base URL")
    parser.add_argument("--token", help="Timbr token (prefer TIMBR_BENCHMARK_TOKEN env var)")
    parser.add_argument("--ontology", help="Optional ontology override")
    parser.add_argument(
        "--mechanism",
        default="full",
        choices=["full", "deterministic", "llm_judge"],
        help="Scoring mechanism",
    )
    parser.add_argument(
        "--execution",
        default="full",
        choices=["full", "generate_sql_only"],
        help="Execution mode",
    )
    parser.add_argument(
        "--number-of-iterations",
        type=int,
        default=1,
        help="Number of iterations for each benchmark question",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable SSL certificate verification",
    )
    parser.add_argument(
        "--is-jwt",
        action="store_true",
        help="Use JWT authentication mode",
    )
    parser.add_argument("--jwt-tenant-id", help="JWT tenant ID")
    parser.add_argument(
        "--questions-json",
        help=(
            "Optional JSON-encoded question subset. "
            "Pass a JSON array of question ID strings (e.g. '[\"Q1\",\"Q3\"]') to run only those "
            "questions from the benchmark definition, or a full questions dict to override the "
            "benchmark definition entirely."
        ),
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        use_deterministic, use_llm_judge = _resolve_mechanism_flags(args.mechanism)
        resolved_token = _resolve_token(args.token)

        if not resolved_token:
            raise ValueError("Token is required. Pass --token or set TIMBR_BENCHMARK_TOKEN/TIMBR_TOKEN.")

        # Resolve optional questions subset
        queries: Optional[Union[Dict[str, Any], List[str]]] = None
        if args.questions_json:
            parsed = json.loads(args.questions_json)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("--questions-json must be a JSON object or a JSON array of question IDs.")
            queries = parsed

        run_benchmark(
            benchmark_name=args.benchmark_name,
            queries=queries,
            url=args.url,
            token=resolved_token,
            ontology=args.ontology,
            use_deterministic=use_deterministic,
            use_llm_judge=use_llm_judge,
            execution=args.execution,
            number_of_iterations=args.number_of_iterations,
            verify_ssl=args.verify_ssl,
            is_jwt=True if args.is_jwt else None,
            jwt_tenant_id=args.jwt_tenant_id,
        )

        logger.info("Benchmark run completed successfully for '%s'.", args.benchmark_name)
        return 0
    except Exception as exc:
        logger.exception("Benchmark run failed: %s", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
