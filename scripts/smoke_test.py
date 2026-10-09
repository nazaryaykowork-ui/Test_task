"""Checks access to BigQuery and to both Gemini models. Run: python -m scripts.smoke_test"""

from agent.bq_client import BigQueryRunner
from agent.config import load_config
from agent.llm import LLM


def main() -> None:
    cfg = load_config()

    bq = BigQueryRunner(project_id=cfg.gcp_project_id, dataset_id=cfg.dataset)
    counts = " UNION ALL ".join(
        f"SELECT '{t}' AS table_name, COUNT(*) AS row_count FROM `{cfg.dataset}.{t}`" for t in cfg.tables
    )
    print("BigQuery:")
    print(bq.execute_query(counts).to_string(index=False))

    # Call each model directly, bypassing the fallback, to check both are reachable.
    client = LLM(cfg).client
    for model in (cfg.model, cfg.fallback_model):
        try:
            reply = client.models.generate_content(model=model, contents="Reply with exactly: OK").text.strip()
        except Exception as e:  # report each model separately; one failing must not hide the other
            reply = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
        print(f"\nGemini {model}: {reply}")


if __name__ == "__main__":
    main()
