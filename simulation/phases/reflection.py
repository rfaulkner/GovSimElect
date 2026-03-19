"""ReflectionPhase — individual reflection at home.

Drives every agent through the environment's ``home`` sub-phase.
Each agent reflects on the round's harvesting events.  The last
agent's env step triggers the round transition (pool regeneration,
termination check, round counter increment).
"""

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

  def execute(self, ctx: PhaseContext) -> PhaseContext:
    if ctx.debug:
      print(
          f"\nROUND {ctx.round_num}: REFLECTION PHASE"
          "\n========================="
      )

    while ctx.env.phase == "home":
      agent = ctx.personas[ctx.agent_id]
      obs = ctx.obs
      sync_agent_state(agent, ctx)

      agent.current_time = obs.current_time
      agent.perceive.perceive(obs)

      agent.reflect.run(["harvesting"])

      if ctx.debug:
        print(f"REFLECT: {agent.identity.name}.")

      action = PersonaAction(agent.agent_id, "home")
      agent.memory.save()

      ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)
      log_step(ctx, action)
      if check_terminated(ctx, termination):
        return ctx

    return ctx
