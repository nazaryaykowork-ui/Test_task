"""Settings from config.yaml, secrets from .env, prompts from prompts.yaml."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    gcp_project_id: str
    model: str
    fallback_model: str
    llm_timeout_s: int
    dataset: str
    tables: tuple[str, ...]
    pii_columns: tuple[str, ...]
    max_bytes_billed: int
    bq_timeout_s: int
    max_steps: int
    max_attempts: int
    max_rows: int
    turn_budget_s: int
    sqlite_path: Path
    log_path: Path
    user_categories: dict[str, tuple[str, ...]]


def load_config(path: Path = ROOT / "config.yaml") -> Config:
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load(path.read_text())
    return Config(
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        gcp_project_id=_require_env("GCP_PROJECT_ID"),
        model=raw["llm"]["model"],
        fallback_model=raw["llm"]["fallback_model"],
        llm_timeout_s=raw["llm"]["timeout_s"],
        dataset=raw["bigquery"]["dataset"],
        tables=tuple(raw["bigquery"]["tables"]),
        pii_columns=tuple(raw["bigquery"]["pii_columns"]),
        max_bytes_billed=raw["bigquery"]["max_bytes_billed"],
        bq_timeout_s=raw["bigquery"]["timeout_s"],
        max_steps=raw["agent"]["max_steps"],
        max_attempts=raw["agent"]["max_attempts"],
        max_rows=raw["agent"]["max_rows"],
        turn_budget_s=raw["agent"]["turn_budget_s"],
        sqlite_path=ROOT / raw["storage"]["sqlite_path"],
        log_path=ROOT / raw["storage"]["log_path"],
        user_categories={name: tuple(u["categories"]) for name, u in raw["users"].items()},
    )


def load_prompts(path: Path = ROOT / "prompts.yaml") -> dict[str, str]:
    """Read on every call, so edits to prompts.yaml apply without a restart."""
    return yaml.safe_load(path.read_text())


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
