"""Factory for creating ElectionSystem instances by name."""

from .base import ElectionSystem
from .fptp import FirstPastThePost
from .proportional import ProportionalRepresentation

# Maps user-facing config strings to concrete ElectionSystem classes.
_REGISTRY: dict[str, type[ElectionSystem]] = {
    "fptp": FirstPastThePost,
    "first_past_the_post": FirstPastThePost,
    "proportional": ProportionalRepresentation,
    "proportional_representation": ProportionalRepresentation,
}


def get_election_system(name: str) -> ElectionSystem:
  """Return an :class:`ElectionSystem` instance for the given name.

  Args:
    name: Case-insensitive election type identifier.  Supported values:
      ``"fptp"``, ``"first_past_the_post"``,
      ``"proportional"``, ``"proportional_representation"``.

  Returns:
    An instantiated :class:`ElectionSystem`.

  Raises:
    ValueError: If ``name`` is not a known election type.
  """
  key = name.strip().lower()
  cls = _REGISTRY.get(key)
  if cls is None:
    supported = ", ".join(sorted(_REGISTRY))
    raise ValueError(
        f"Unknown election type '{name}'. Supported types: {supported}"
    )
  return cls()
