from typing import Protocol

# controlls overall types for models


class HasModelAttribute(Protocol):
    index_set: list  # list of all indicies [i,j,k, ...]
    neighbors: list  # list of linked indicies [(i,j) ...]
