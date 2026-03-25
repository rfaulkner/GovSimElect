"""DiscussionPhase — group conversation at the restaurant.

Drives the single ``restaurant`` env step where agents hold a
group conversation.  The converse component handles turn-taking
internally; the env advances all agents to ``home`` once the
conversation action is processed.
"""

from simulation.persona.common import PersonaActionChat

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext
from simulation.phases.base import check_terminated
from simulation.phases.base import log_step
from simulation.phases.base import sync_agent_state


class DiscussionPhase(Phase):
  """Agents converse in a group discussion at the restaurant."""

  @property
  def name(self) -> str:
    return "discussion"

  async def execute(self, ctx: PhaseContext) -> PhaseContext:
    if ctx.debug:
      print(
          f"\nROUND {ctx.round_num}: DISCUSSION PHASE"
          "\n========================="
      )

    while ctx.env.phase == "restaurant":
      agent = ctx.personas[ctx.agent_id]
      obs = ctx.obs
      sync_agent_state(agent, ctx)

      agent.current_time = obs.current_time
      agent.perceive.perceive(obs)

      other_personas = []
      for aid, location in obs.current_location_agents.items():
        if location == "restaurant":
          other_personas.append(
              agent.other_personas_from_id[aid]
          )

      (
          conversation,
          _,
          resource_limit,
          html_interactions,
      ) = agent.converse.converse_group(
          other_personas,
          obs.current_location,
          obs.current_time,
          obs.context,
          obs.agent_resource_num,
          mayoral_agenda=agent.agenda,
          harvest_report=agent.harvest_report,
          leader_persona=agent.current_leader,
          debug=ctx.debug,
      )

      action = PersonaActionChat(
          agent.agent_id,
          "restaurant",
          conversation,
          conversation_resource_limit=resource_limit,
          stats={"conversation_resource_limit": resource_limit},
          html_interactions=html_interactions,
      )

      agent.memory.save()

      ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)
      log_step(ctx, action)
      if check_terminated(ctx, termination):
        return ctx

    return ctx
