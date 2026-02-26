"""Election system interface and implementations for the fishing simulation."""

from .base import ElectionSystem
from .factory import get_election_system
from .fptp import FirstPastThePost
from .proportional import ProportionalRepresentation

__all__ = [
    "ElectionSystem",
    "FirstPastThePost",
    "ProportionalRepresentation",
    "get_election_system",
]
