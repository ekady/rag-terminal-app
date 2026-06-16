from langchain_text_splitters import (
    CharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config import Config


def fixed_size_overlap(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )
    return splitter.split_text(text)


def sentence_based(
    text: str,
    chunk_size: int = None,
) -> list[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE

    splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
    )
    return splitter.split_text(text)


def recursive(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def semantic(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    return splitter.split_text(text)


STRATEGIES = {
    "fixed": fixed_size_overlap,
    "sentence": sentence_based,
    "recursive": recursive,
    "semantic": semantic,
}


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}"
        )

    fn = STRATEGIES[strategy]
    if strategy == "sentence":
        return fn(text, chunk_size=chunk_size)
    return fn(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
