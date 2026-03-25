"""ReflectionPhase — individual reflection at home.

Drives every agent through the environment's ``home`` sub-phase.
Each agent reflects on the round's harvesting events.  The last
agent's env step triggers the round transition (pool regeneration,
termination check, round counter increment).

All reflection LLM calls run concurrently via ``asyncio.gather``;
environment stepping remains sequential.
"""

import asyncio

from simulation.persona.common import PersonaAction

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext
from simulation.phases.base import check_terminated
from simulation.phases.base import log_step
from simulation.phases.base import sync_agent_state


class ReflectionPhase(Phase):
  """Each agent reflects on the round at home."""

  @property
  def name(self) -> str:
    return "reflection"

  async def execute(self, ctx: PhaseContext) -> PhaseContext:
    if ctx.debug:
      print(
          f"\nROUND {ctx.round_num}: REFLECTION PHASE"
          "\n========================="
      )

    # ── Prepare all agents ─────────────────────────────────────────
    home_agents = []
    for agent_id in ctx.env.agents:
      agent = ctx.personas[agent_id]
      sync_agent_state(agent, ctx)
      agent.current_time = ctx.obs.current_time
      agent.perceive.perceive(ctx.obs)
      home_agents.append(agent)

    if not home_agents:
      return ctx

    # ── Gather all reflection LLM calls concurrently ───────────────
    async def one_reflect(agent):
      await agent.reflect.arun(["harvesting"])

    await asyncio.gather(
        *(one_reflect(agent) for agent in home_agents)
    )

    if ctx.debug:
      for agent in home_agents:
        print(f"REFLECT: {agent.identity.name}.")

    # ── Step env sequentially for every agent ──────────────────────
    for agent in home_agents:
      action = PersonaAction(agent.agent_id, "home")
      agent.memory.save()
      ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)
      log_step(ctx, action)
      if check_terminated(ctx, termination):
        return ctx

    return ctx

