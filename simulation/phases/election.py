"""ElectionPhase — agents vote on leader agendas.

Uses the agendas produced by PolicyMakingPhase.  Each non-leader
persona casts a vote; the winner's agenda becomes the round's
guiding policy.  Election memories are stored for all agents.
"""

import asyncio
import random

from simulation.environment import concurrent_env

from simulation.persona.cognition import leaders as leaders_lib
from simulation.persona.common import PersonaEvent

from simulation.phases.base import Phase
from simulation.phases.base import PhaseContext


class ElectionPhase(Phase):
  """Agents vote on leader agendas and a winner is determined."""

  @property
  def name(self) -> str:
    return "election"

  async def execute(self, ctx: PhaseContext) -> PhaseContext:
    if not ctx.leader_candidates:
      return ctx

    leader_agendas = ctx.leader_agendas
    if leader_agendas is None:
      # No agendas available (policy_making didn't run).
      return ctx

    if ctx.debug:
      print(
          f"\n\nROUND {ctx.round_num}: ELECTION"
          "\n=================="
      )

    votes = {
        leader.identity.name: 0
        for leader in ctx.leader_candidates.values()
    }

    if len(ctx.leader_candidates) > 1:
      # ── Gather all votes concurrently ──────────────────────────
      async def one_vote(persona_id):
        retrieved_memory = leaders_lib.get_memories(
            ctx.personas[persona_id]
        )
        candidates = [
            leader.identity.name
            for _, leader in ctx.leader_candidates.items()
        ]
        random.shuffle(candidates)
        vote, _ = await ctx.personas[
            persona_id
        ].act.aparticipate_in_election(
            retrieved_memories=retrieved_memory,
            current_location="",
            current_time=ctx.obs.current_time.strftime(
                "%H-%M-%S"
            ),
            candidates=candidates,
            leader_agendas=leader_agendas,
            debug=ctx.debug,
        )
        return persona_id, vote

      voter_ids = [
          pid for pid in ctx.personas
          if pid not in ctx.leader_candidates
      ]
      results = await asyncio.gather(
          *(one_vote(pid) for pid in voter_ids)
      )

      for persona_id, vote in results:
        candidate_id = (
            vote.name if hasattr(vote, "name") else str(vote)
        )
        candidate_str = ctx.agent_id_to_name.get(
            candidate_id, candidate_id
        )
        votes[candidate_str] = votes.get(candidate_str, 0) + 1
        ctx.personas[persona_id].store.store_event(
            PersonaEvent(
                f"Round {ctx.round_num} vote: {vote}",
                created=ctx.obs.current_time,
                expiration=concurrent_env.get_expiration_next_month(
                    ctx.obs.current_time
                ),
                always_include=True,
            )
        )

      votes_cp = dict(votes)
      if "none" in votes_cp:
        del votes_cp["none"]
      max_votes = max(votes_cp.values())
      keys = [
          key for key, value in votes_cp.items()
          if value == max_votes
      ]
      winner = random.choice(keys)
    elif len(ctx.leader_candidates) == 1:
      print("SKIPPING ELECTION AS ONLY ONE LEADER CANDIDATE...")
      winner_key = list(ctx.leader_candidates.keys())[0]
      winner = ctx.agent_id_to_name[winner_key]
    else:
      raise ValueError("No leader candidates.")

    if ctx.debug:
      print(
          "\n=================\nELECTION RESULTS"
          "\n================="
      )
      for candidate, vote_count in votes.items():
        print(f"{candidate}: {vote_count} votes")
      print(f"\nROUND {ctx.round_num} WINNER: {winner}")
      print(
          "\n=================\nLEADER AGENDAS"
          "\n================="
      )
      for agenda_id, agenda_text in leader_agendas.items():
        print(
            f"\n{ctx.agent_id_to_name.get(agenda_id, agenda_id)}'s"
            " Agenda:\n=================="
        )
        pid = ctx.agent_name_to_id.get(agenda_id, agenda_id)
        if pid in ctx.leader_candidates:
          print(
              "SVO Angle:"
              f" {ctx.leader_candidates[pid].svo_angle},"
              " SVO Type:"
              f" {ctx.leader_candidates[pid].svo_type}\n"
          )
        print(agenda_text)

    leader_agendas["none"] = (
        "No leader agenda, use your best judgement."
    )
    leader_announcement = (
        f"Newly elected leader {winner}'s round"
        f" {ctx.round_num} agenda:"
        f" {leader_agendas[winner]}"
    )
    leaders_lib.make_public_leader_memories(
        all_personas=ctx.personas,
        leader_announcement=leader_announcement,
        current_time=ctx.obs.current_time,
    )

    ctx.winner = winner
    ctx.votes = votes
    ctx.agenda = leader_agendas[winner]
    return ctx

