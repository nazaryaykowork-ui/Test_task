"""Interactive chat. Run: python -m agent.cli --user alice [--thread ID]"""

import argparse
import logging
import sys
import uuid

from google.auth.exceptions import DefaultCredentialsError

from agent.chat import Chat
from agent.config import ConfigError, load_config
from agent.graph import can_access, open_graph
from agent.pii import mask_pii
from agent.resilience import ServiceUnavailable
from agent.tracing import summarize

NO_CREDENTIALS = (
    "Cannot start: no Google Cloud credentials. Run `gcloud auth application-default login`, "
    "or set GOOGLE_APPLICATION_CREDENTIALS in .env to a service-account key file."
)


class MaskingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return mask_pii(super().format(record))


def main() -> None:
    parser = argparse.ArgumentParser(description="Data analysis chat agent")
    parser.add_argument("--user", required=True, help="user id from config.yaml")
    parser.add_argument("--thread", help="resume an existing conversation")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as e:
        sys.exit(f"Cannot start: {e}")
    setup_logging(cfg.log_path.with_name("python.log"))
    if args.user not in cfg.user_categories:
        sys.exit(f"Unknown user '{args.user}'. Known users: {', '.join(cfg.user_categories)}")

    try:
        with open_graph(cfg) as (graph, tracer, budget):
            # Unknown and foreign threads get the same message, so the CLI does not reveal
            # which thread ids exist.
            if args.thread and not can_access(graph, args.thread, args.user):
                sys.exit(f"Thread '{args.thread}' not found for user '{args.user}'.")
            chat_loop(Chat(graph, tracer, args.user, args.thread or str(uuid.uuid4()), budget))
    except DefaultCredentialsError:
        sys.exit(NO_CREDENTIALS)
    except ServiceUnavailable as e:  # the table schemas are loaded at startup
        sys.exit(f"Cannot start: {e}")
    except KeyboardInterrupt:  # Ctrl-C during startup
        sys.exit("\nStopped.")


def setup_logging(path) -> None:
    """Library logs (the BigQuery client logs every failed query) go to a file, not the chat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(MaskingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.captureWarnings(True)


def chat_loop(chat: Chat) -> None:
    print(f"user: {chat.user_id} | thread: {chat.thread_id}")
    print("Ask a question about the sales data. /stats for agent metrics, /exit to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
            if text == "/exit":
                break
            if text == "/stats":
                print(summarize(chat.tracer.path) + "\n")
            elif text:
                print(f"agent> {chat.send(text)}\n")
        except (EOFError, KeyboardInterrupt):
            break
    print(f"\nBye. Resume this conversation with --thread {chat.thread_id}")


if __name__ == "__main__":
    main()
