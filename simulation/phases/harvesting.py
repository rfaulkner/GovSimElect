"""HarvestingPhase — agents fish and harvest report is generated.

Drives every agent through the environment's ``lake`` and
``pool_after_harvesting`` sub-phases.  After all agents have
harvested, the leader's harvest report is generated and broadcast
as public memories.

All harvest LLM decisions run concurrently via ``asyncio.gather``
since the ``ConcurrentEnv`` collects every agent's desired catch
before assigning resources proportionally.
"""

import asyncio

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

  async def execute(self, ctx: PhaseContext) -> PhaseContext:
    if ctx.debug:
      print(
          f"\nROUND {ctx.round_num}: HARVESTING PHASE"
          "\n========================="
      )

    # ── Lake sub-phase: all agents decide concurrently ─────────────
    # In ConcurrentEnv every agent sees the same pool state — the
    # resource count only changes once all desired catches are in.
    # So we gather all LLM decisions in parallel, then step the env
    # sequentially to feed the collected actions.
    initial_obs = ctx.obs  # Same observation for every agent.

    # Prepare every agent and retrieve their memories.
    lake_agent_data = []
    for agent_id in ctx.env.agents:
      agent = ctx.personas[agent_id]
      sync_agent_state(agent, ctx)
      agent.current_time = initial_obs.current_time
      agent.perceive.perceive(initial_obs)

      retrieved_memory = agent.retrieve.retrieve(
          [initial_obs.current_location], 10
      )
      if ctx.debug:
        str_memory = "\n".join(str(m) for m in retrieved_memory)
        print(f"MEMORIES {agent.identity.name}:\n{str_memory}")

      lake_agent_data.append((agent_id, agent, retrieved_memory))

    # ── Gather all harvest LLM calls concurrently ──────────────────
    async def one_harvest(agent_id, agent, retrieved_memory):
      if initial_obs.current_resource_num > 0:
        num_resource, html_interactions = (
            await agent.act.achoose_how_many_fish_to_catch(
                retrieved_memory,
                initial_obs.current_location,
                initial_obs.current_time,
                initial_obs.context,
                range(0, initial_obs.current_resource_num + 1),
                initial_obs.before_harvesting_sustainability_threshold,
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

      return agent_id, agent, action

    harvest_results = await asyncio.gather(
        *(one_harvest(aid, a, rm) for aid, a, rm in lake_agent_data)
    )

    # ── Step env sequentially with the gathered harvest actions ─────
    for _, agent, action in harvest_results:
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
      harvest_report = await leaders_lib.amake_leader_report(
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

