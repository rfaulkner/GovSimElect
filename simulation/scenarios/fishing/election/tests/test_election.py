"""Unit tests for the election system package."""

import random
import unittest
from collections import Counter

from simulation.scenarios.fishing.election import (
    ElectionSystem,
    FirstPastThePost,
    ProportionalRepresentation,
    get_election_system,
)


class TestFirstPastThePost(unittest.TestCase):

  def setUp(self):
    self.system = FirstPastThePost()

  def test_clear_winner(self):
    votes = {"Alice": 5, "Bob": 3, "Carol": 1}
    self.assertEqual(self.system.determine_winner(votes), "Alice")

  def test_clear_winner_zero_for_others(self):
    votes = {"Alice": 1, "Bob": 0}
    self.assertEqual(self.system.determine_winner(votes), "Alice")

  def test_tie_broken_randomly(self):
    """Both tied candidates should be reachable across many draws."""
    votes = {"Alice": 3, "Bob": 3}
    random.seed(42)
    outcomes = {self.system.determine_winner(votes) for _ in range(50)}
    self.assertIn("Alice", outcomes)
    self.assertIn("Bob", outcomes)

  def test_three_way_tie(self):
    votes = {"Alice": 2, "Bob": 2, "Carol": 2}
    winner = self.system.determine_winner(votes)
    self.assertIn(winner, {"Alice", "Bob", "Carol"})

  def test_single_candidate(self):
    votes = {"Alice": 7}
    self.assertEqual(self.system.determine_winner(votes), "Alice")

  def test_empty_votes_raises(self):
    with self.assertRaises(ValueError):
      self.system.determine_winner({})


class TestProportionalRepresentation(unittest.TestCase):

  def setUp(self):
    self.system = ProportionalRepresentation()

  def test_returns_a_candidate(self):
    votes = {"Alice": 4, "Bob": 2, "Carol": 1}
    winner = self.system.determine_winner(votes)
    self.assertIn(winner, votes)

  def test_distribution_roughly_proportional(self):
    """Over many draws, winner frequency should mirror vote share."""
    votes = {"Alice": 6, "Bob": 3, "Carol": 1}
    total = sum(votes.values())
    random.seed(0)
    trials = 10_000
    counts = Counter(self.system.determine_winner(votes) for _ in range(trials))

    for candidate, vote_count in votes.items():
      expected_proportion = vote_count / total
      actual_proportion = counts[candidate] / trials
      # Allow ±5 percentage points tolerance.
      self.assertAlmostEqual(
          actual_proportion, expected_proportion, delta=0.05,
          msg=f"{candidate}: expected ~{expected_proportion:.2%}, got {actual_proportion:.2%}",
      )

  def test_zero_votes_falls_back_to_uniform(self):
    """If all candidates have 0 votes, any candidate can win."""
    votes = {"Alice": 0, "Bob": 0}
    random.seed(1)
    outcomes = {self.system.determine_winner(votes) for _ in range(40)}
    self.assertIn("Alice", outcomes)
    self.assertIn("Bob", outcomes)

  def test_single_candidate(self):
    votes = {"Alice": 5}
    self.assertEqual(self.system.determine_winner(votes), "Alice")

  def test_empty_votes_raises(self):
    with self.assertRaises(ValueError):
      self.system.determine_winner({})


class TestGetElectionSystem(unittest.TestCase):

  def test_factory_fptp(self):
    system = get_election_system("fptp")
    self.assertIsInstance(system, FirstPastThePost)

  def test_factory_fptp_long_name(self):
    system = get_election_system("first_past_the_post")
    self.assertIsInstance(system, FirstPastThePost)

  def test_factory_proportional(self):
    system = get_election_system("proportional")
    self.assertIsInstance(system, ProportionalRepresentation)

  def test_factory_proportional_long_name(self):
    system = get_election_system("proportional_representation")
    self.assertIsInstance(system, ProportionalRepresentation)

  def test_factory_case_insensitive(self):
    self.assertIsInstance(get_election_system("FPTP"), FirstPastThePost)
    self.assertIsInstance(
        get_election_system("Proportional"), ProportionalRepresentation
    )

  def test_factory_whitespace_tolerant(self):
    self.assertIsInstance(get_election_system("  fptp  "), FirstPastThePost)

  def test_factory_unknown_raises(self):
    with self.assertRaises(ValueError, msg="Should raise for unknown type"):
      get_election_system("ranked_choice")

  def test_factory_returns_election_system_subclass(self):
    for name in ("fptp", "proportional"):
      with self.subTest(name=name):
        self.assertIsInstance(get_election_system(name), ElectionSystem)


if __name__ == "__main__":
  unittest.main()
