"""Proportional Representation (PR) election system.

Because the simulation has a single leader slot, Proportional Representation
is modelled as a *weighted random selection*: each candidate's probability of
winning equals their share of the total votes cast.  This is consistent with
the spirit of PR (outcomes reflect vote proportions) while remaining compatible
with a single-winner setup.

Reference: https://en.wikipedia.org/wiki/Proportional_representation
"""

import random

from .base import ElectionSystem


class ProportionalRepresentation(ElectionSystem):
  """Select a winner with probability proportional to vote share.

  If all candidates receive zero votes (e.g. no non-leader voters participated)
  then all candidates are treated as equally likely (uniform distribution).
  """

  def determine_winner(self, votes: dict[str, int]) -> str:
    """Return a candidate drawn with probability proportional to votes.

    Args:
      votes: Mapping of candidate name → vote count (``"none"`` excluded).

    Returns:
      Name of the winning candidate.

    Raises:
      ValueError: If ``votes`` is empty.
    """
    if not votes:
      raise ValueError("Cannot determine a winner from an empty vote tally.")

    candidates = list(votes.keys())
    weights = [votes[c] for c in candidates]
    total = sum(weights)

    if total == 0:
      # No votes cast — fall back to a uniform draw so a winner is still
      # chosen (mirrors FPTP tie-breaking behaviour).
      return random.choice(candidates)

    # random.choices supports weighted selection natively.
    return random.choices(candidates, weights=weights, k=1)[0]
