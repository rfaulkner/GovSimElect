"""PolicyMakingPhase — leaders generate their policy agendas.

Runs before elections so that voters can evaluate platforms.
Each leader candidate crafts an agenda influenced by their SVO,
past harvest data, and the previous winning agenda (if any).
"""

from simulation.persona.cognition import leaders as leaders_lib

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext


class PolicyMakingPhase(Phase):
  """Each leader candidate crafts a policy agenda for the upcoming round."""

  @property
  def name(self) -> str:
    return "policy_making"

  def execute(self, ctx: PhaseContext) -> PhaseContext:
    if not ctx.leader_candidates:
      return ctx

    if ctx.debug:
      print(
          f"\n\nROUND {ctx.round_num}: POLICY MAKING"
          "\n=================="
      )

    leader_agendas = {}
    for _, leader in ctx.leader_candidates.items():
      agenda, _ = leaders_lib.prompt_leader_agenda(
          model=ctx.wrapper,
          init_persona=leader,
          current_location="restaurant",
          current_time=ctx.obs.current_time,
          init_retrieved_memory=leaders_lib.get_memories(leader),
          total_fishers=len(ctx.personas),
          svo_angle=leader.svo_angle,
          last_winning_agenda=(
              ctx.agenda if ctx.round_num > 0 else None
          ),
          harvest_report=ctx.harvest_report,
          harvest_stats=(
              ctx.round_harvest_stats.get(ctx.round_num - 1)
              if ctx.round_num > 0
              else None
          ),
          use_disinfo=ctx.disinformation,
      )
      leader_agendas[leader.identity.name] = agenda

    ctx.leader_agendas = leader_agendas
    return ctx
