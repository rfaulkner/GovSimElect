"""Abstract base class for election systems."""

import abc


class ElectionSystem(abc.ABC):
  """Interface for election winner-determination logic.

  Subclasses implement :meth:`determine_winner` to encapsulate a specific
  electoral rule.  The rest of the simulation (vote collection, memory
  storage, etc.) is handled by :func:`perform_election` in ``run_election.py``
  and is independent of which system is active.
  """

  @abc.abstractmethod
  def determine_winner(self, votes: dict[str, int]) -> str:
    """Determine the election winner from a vote tally.

    Args:
      votes: Mapping of candidate name → number of votes received.
        Candidates with zero votes may be present.  The special key
        ``"none"`` (if present) should be excluded from consideration
        before calling this method.

    Returns:
      The name of the winning candidate.

    Raises:
      ValueError: If ``votes`` is empty or no winner can be determined.
    """
