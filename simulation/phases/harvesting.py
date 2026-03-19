"""HarvestingPhase — agents fish and harvest report is generated.

Drives every agent through the environment's ``lake`` and
``pool_after_harvesting`` sub-phases.  After all agents have
harvested, the leader's harvest report is generated and broadcast
as public memories.
"""

from simulation.persona.cognition import leaders as leaders_lib
from simulation.persona.common import PersonaAction
from simulation.persona.common import PersonaActionHarvesting

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext
from simulation.phases.base import check_terminated
from simulation.phases.base import log_step
from simulation.phases.base import sync_agent_state


class HarvestingPhase(Phase):
  """Agents decide how many fish to catch; harvest report generated after."""

  @property
  def name(self) -> str:
    return "harvesting"

  def execute(self, ctx: PhaseContext) -> PhaseContext:
    if ctx.debug:
      print(
          f"\nROUND {ctx.round_num}: HARVESTING PHASE"
          "\n========================="
      )

    # ── Lake sub-phase: each agent decides how many fish to catch ──
    while ctx.env.phase == "lake":
      agent = ctx.personas[ctx.agent_id]
      obs = ctx.obs
      sync_agent_state(agent, ctx)

      agent.current_time = obs.current_time
      agent.perceive.perceive(obs)

      retrieved_memory = agent.retrieve.retrieve(
          [obs.current_location], 10
      )
      if ctx.debug:
        str_memory = "\n".join(str(m) for m in retrieved_memory)
        print(f"MEMORIES {agent.identity.name}:\n{str_memory}")

      if obs.current_resource_num > 0:
        num_resource, html_interactions = (
            agent.act.choose_how_many_fish_to_catch(
                retrieved_memory,
                obs.current_location,
                obs.current_time,
                obs.context,
                range(0, obs.current_resource_num + 1),
                obs.before_harvesting_sustainability_threshold,
                agent.agenda,
                debug=ctx.debug,
            )
        )
        action = PersonaActionHarvesting(
            agent.agent_id,
            "lake",
            num_resource,
            stats={
                f"{agent.agent_id}_collected_resource": num_resource,
            },
            html_interactions=html_interactions,
        )
      else:
        num_resource = 0
        action = PersonaActionHarvesting(
            agent.agent_id,
            "lake",
            num_resource,
            stats={},
            html_interactions=(
                "<strong>Framework<strong/>: no fish to catch"
            ),
        )

      if ctx.debug:
        print(f"HARVEST: {agent.identity.name} {num_resource}.")

      agent.memory.save()

      ctx.round_harvest_stats[ctx.round_num][
          agent.identity.name
      ] = action.quantity

      ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)
      log_step(ctx, action)
      if check_terminated(ctx, termination):
        return ctx

    # ── Pool-after-harvesting sub-phase ────────────────────────────
    while ctx.env.phase == "pool_after_harvesting":
      agent = ctx.personas[ctx.agent_id]
      obs = ctx.obs

      agent.current_time = obs.current_time
      agent.perceive.perceive(obs)

      action = PersonaAction(agent.agent_id, "lake")
      agent.memory.save()

      ctx.agent_id, ctx.obs, _, termination = ctx.env.step(action)
      log_step(ctx, action)
      if check_terminated(ctx, termination):
        return ctx

    # ── Record round stats (post-harvest pool state) ───────────────
    ctx.round_stats[ctx.round_num] = {
        "num_resources": ctx.env.internal_global_state[
            "resource_in_pool"
        ],
        "regen_factor": ctx.env.internal_global_state["regen_factor"],
    }
    print(f"ROUND {ctx.round_num} ROUND STATS: {ctx.round_stats}")

    # ── Generate harvest report and broadcast public memories ──────
    if ctx.leader_candidates:
      assert ctx.winner is not None
      harvest_report = leaders_lib.make_leader_report(
          personas=ctx.personas,
          leader_candidates=ctx.leader_candidates,
          current_time=ctx.obs.current_time,
          wrapper=ctx.wrapper,
          disinformation=ctx.disinformation,
          agenda=ctx.agenda,
          curr_round=ctx.round_num,
          winner_id=ctx.agent_name_to_id[ctx.winner],
          round_harvest_stats=ctx.round_harvest_stats[ctx.round_num],
          regen_factor=ctx.env.internal_global_state["regen_factor"],
          debug=ctx.debug,
      )
      announcement = (
          f"{ctx.winner}'s ROUND {ctx.round_num} REPORT:"
          f" {harvest_report}"
      )
      leaders_lib.make_public_leader_memories(
          all_personas=ctx.personas,
          leader_announcement=announcement,
          current_time=ctx.obs.current_time,
      )
    else:
      print("NO LEADER CANDIDATES - MAKING FACTUAL REPORT ...")
      harvest_report = leaders_lib.make_harvest_report(
          ctx.personas, ctx.round_harvest_stats[ctx.round_num]
      )
    ctx.harvest_report = harvest_report

    return ctx
