"""Small deterministic fixtures; real model downloads require explicit opt-in."""

from collections.abc import Callable

import pytest

from ragstat.ingestion.chunker import Chunk


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration", action="store_true", help="Allow real model integration tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Use --run-integration to load real model weights")
                )


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    def create(text: str, identifier: str = "a", document_id: str | None = None) -> Chunk:
        return Chunk(
            id=identifier,
            document_id=document_id or identifier,
            text=text,
            start_word=0,
            end_word=len(text.split()),
        )

    return create
