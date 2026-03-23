"""Base phase abstraction and shared context for simulation phases.

Provides the Phase ABC that all simulation phases implement, the
PhaseContext dataclass that carries mutable state between phases,
and small helper functions used by multiple phases.
"""

import abc
import collections
import dataclasses
from typing import Any

import numpy as np

from simulation.persona.persona import DEFAULT_AGENDA


@dataclasses.dataclass
class PhaseContext:
  """Mutable state shared between phases within and across rounds."""

  # ── Immutable configuration ──────────────────────────────────────
  cfg: Any  # omegaconf.DictConfig (experiment-level)
  personas: dict
  leader_candidates: dict
  env: Any
  wrapper: Any  # ModelWandbWrapper — agent LLM calls
  logger: Any  # WandbLogger — game metric logging
  agent_id_to_name: dict[str, str]
  agent_name_to_id: dict[str, str]
  experiment_storage: str
  consolidated_log_path: str
  disinformation: bool
  debug: bool

  # ── Current env stepping state ───────────────────────────────────
  agent_id: str = ""
  obs: Any = None

  # ── Per-round mutable state (persists across rounds) ─────────────
  round_num: int = 0
  winner: str | None = None
  votes: dict | None = None
  leader_agendas: dict | None = None
  agenda: str = dataclasses.field(default=DEFAULT_AGENDA)
  harvest_report: str | None = None
  round_harvest_stats: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(
          lambda: collections.defaultdict(int)
      )
  )
  round_stats: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(dict)
  )
  election_results: dict = dataclasses.field(default_factory=dict)
  terminated: bool = False


class Phase(abc.ABC):
  """A single simulation phase that runs once per round."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Short identifier for this phase, e.g. 'election'."""

  @abc.abstractmethod
  async def execute(self, ctx: PhaseContext) -> PhaseContext:
    """Run the phase, mutating and returning the context."""


# ── Helpers shared by multiple phase implementations ─────────────────


def sync_agent_state(agent, ctx: PhaseContext) -> None:
  """Update an agent's transient state from the shared context."""
  agent.update_agenda(ctx.agenda)
  agent.update_harvest_report(ctx.harvest_report)
  if ctx.winner:
    agent.update_current_leader(
        ctx.leader_candidates[ctx.agent_name_to_id[ctx.winner]]
    )


def log_step(ctx: PhaseContext, action) -> None:
  """Log game stats after an env.step() call.

  Must be called *after* ctx.obs has been updated by env.step().
  """
  stats = {}
  if hasattr(action, "stats"):
    for s in [
        "conversation_resource_limit",
        *[
            f"persona_{i}_collected_resource"
            for i in range(ctx.cfg.env.num_agents)
        ],
    ]:
      if s in action.stats:
        stats[s] = action.stats[s]
  ctx.logger.log_game({"num_resource": ctx.obs.current_resource_num, **stats})


def check_terminated(ctx: PhaseContext, termination: dict) -> bool:
  """Check termination flags and log the final step if the sim ends."""
  if np.any(list(termination.values())):
    ctx.logger.log_game(
        {"num_resource": ctx.obs.current_resource_num}, last_log=True
    )
    ctx.terminated = True
    return True
  return False
