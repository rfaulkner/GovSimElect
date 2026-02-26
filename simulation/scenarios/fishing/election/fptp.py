"""First Past the Post (FPTP) election system."""

import random

from .base import ElectionSystem


class FirstPastThePost(ElectionSystem):
  """Winner is the candidate with the most votes.

  Ties are broken by uniform random selection among all tied candidates.
  This preserves the original behaviour of ``perform_election``.
  """

  def determine_winner(self, votes: dict[str, int]) -> str:
    """Return the candidate with the most votes, breaking ties randomly.

    Args:
      votes: Mapping of candidate name → vote count (``"none"`` excluded).

    Returns:
      Name of the winning candidate.

    Raises:
      ValueError: If ``votes`` is empty.
    """
    if not votes:
      raise ValueError("Cannot determine a winner from an empty vote tally.")
    max_votes = max(votes.values())
    winners = [name for name, count in votes.items() if count == max_votes]
    return random.choice(winners)
