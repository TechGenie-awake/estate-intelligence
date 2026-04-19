from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
