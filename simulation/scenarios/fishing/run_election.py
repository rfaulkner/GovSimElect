"""Runs simulation with an election."""

import collections
import datetime
import json
import os
import random

import numpy as np
import omegaconf

from simulation.persona import EmbeddingModel
from simulation.persona import PersonaAgent
from simulation.persona import SVOPersonaType

from simulation.persona.common import PersonaActionHarvesting
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .agents.persona_v3 import FishingPersona
from .agents.persona_v3.cognition import leaders as leaders_lib
from .agents.persona_v3.cognition import utils as cognition_utils

from .environment import FishingConcurrentEnv
from .environment import FishingPerturbationEnv


cognition_utils.SYS_VERSION = "v3"


def get_memories(
    persona: PersonaAgent, current_location: str = "lake") -> list[str]:
  """Get memories for a persona."""
  retireved_memory = []
  try:
    retireved_memory = persona.retrieve.retrieve(
        [current_location], 10)
  except ValueError as e:
    print(f"Couldn't retrieve memories for {persona.identity.name}: {e}")
  return retireved_memory


def perform_election(
    personas: dict[str, PersonaAgent],
    # TODO(rfaulk): pass in sampled leader candidates....
    leader_candidates: dict[str, PersonaAgent],
    current_time: datetime,
    wrapper: ModelWandbWrapper,
    agent_id_to_name: dict[int, str],
    agent_name_to_id: dict[str, int],
    # TODO(rfaulk): Need to fix this or remove, never changes.
    current_location: str = "lake",
    last_winning_agenda: str | None = None,
    harvest_report: str | None = None,
    harvest_stats: str | None = None,
):
  """Runs an election among the leaders."""
  leader_agendas = {}
  # Get updated leader agendas using the leader prompt functions
  for _, leader in leader_candidates.items():
    agenda, _ = leaders_lib.prompt_leader_agenda(
        model=wrapper,
        init_persona=leader,
        current_location=current_location,
        current_time=current_time,
        init_retrieved_memory=get_memories(leader),
        total_fishers=len(personas),
        svo_angle=leader.svo_angle,
        last_winning_agenda=last_winning_agenda,
        harvest_report=harvest_report,
        harvest_stats=harvest_stats,
        use_disinfo=False,  # TODO(rfaulk): Add disinfo options to the election.
    )
    # print(f"\nAGENDA: {leader_candidates[pid].identity.name} {agenda}")
    leader_agendas[leader.identity.name] = agenda

  # print(f"Leader Agendas:\n{leader_agendas}")
  votes = {}
  for persona_id in personas:
    # Only non-leader personas cast votes
    if persona_id not in leader_candidates:
      # Get memories.
      current_location = "lake"
      retireved_memory = get_memories(personas[persona_id], current_location)
      vote, _ = personas[persona_id].act.participate_in_election(
          retireved_memory,  # retrieved memories; adjust as needed
          current_location,  # default location
          current_time.strftime("%H-%M-%S"),  # current time as string
          [leader.identity.name for _, leader in leader_candidates.items()],
          leader_agendas,
      )
      # Determine candidate identifier: if vote has attribute 'name', use it;
      # otherwise, use its string
      candidate_id = vote.name if hasattr(vote, "name") else str(vote)
      # Use the mapping to get the human-readable name; if not found, use
      # candidate_id as is.
      candidate_str = agent_id_to_name.get(candidate_id, candidate_id)
      votes[candidate_str] = votes.get(candidate_str, 0) + 1

  # Determine winner (as the candidate's human-readable name)
  # Randonly break ties.
  winner = max(votes.values())
  keys = [key for key, value in votes.items() if value == winner]
  winner = random.choice(keys)

  print("\nElection Voting Results:")
  for candidate, vote_count in votes.items():
    print(f"{candidate}: {vote_count} votes")
  print(f"\nWinner: {winner}")
  print("\nLeader Agendas:")
  for agenda_id, agenda in leader_agendas.items():
    # Convert leader id to human-readable name using mapping
    print(f"\n{agent_id_to_name.get(agenda_id, agenda_id)}'s Agenda:")
    pid = agent_name_to_id.get(agenda_id, agenda_id)
    print(
        f"SVO Angle: {leader_candidates[pid].svo_angle}, SVO Type:"
        f" {leader_candidates[pid].svo_type}\n"
    )
    print(agenda)
  # In case the voters decide not to vote.
  leader_agendas["none"] = "No leader agenda, use your best judgement."
  return winner, votes, leader_agendas


def run(
    cfg: omegaconf.DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
  """Run the simulatiuon."""
  if not os.path.exists(experiment_storage):
    os.makedirs(experiment_storage)

  # Create a consolidated log file
  consolidated_log_path = os.path.join(
      experiment_storage, "consolidated_results.json"
  )

  # Helper function to append to the consolidated log
  def log_to_file(log_type, data):
    with open(consolidated_log_path, "a") as f:
      entry = {
          "timestamp": datetime.datetime.now().isoformat(),
          "type": log_type,
          "data": data,
      }
      f.write(json.dumps(entry) + "\n")

  # Initialize the log file with a header
  log_to_file(
      "initialization",
      {
          "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
          "config": (
              omegaconf.OmegaConf.to_container(cfg)
              if hasattr(cfg, "to_container")
              else str(cfg)
          ),
      },
  )

  # Stores elections results.
  election_results = {}

  if cfg.agent.agent_package == "persona_v3":
    cognition_utils.REASONING = cfg.agent.cot_prompt
  else:
    raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

  # Get the leader candidates.
  # TODO(rfaulk): Remove hardcode, use config and sample leader populations.
  # leader_distribution = extract_leader_group(cfg.agent.leader_population_type)
  leader_distribution = leaders_lib.LeaderPopulationType.BALANCED
  leader_svos, leader_types = leaders_lib.sample_leader_svos(
      leader_distribution)

  # Initialize leader candidates
  leader_candidates = {}
  for i, svo_data in enumerate(zip(leader_svos, leader_types)):
    svo_angle, svo_type = svo_data
    leader_candidates[f"persona_{i}"] = FishingPersona(
        cfg.agent,
        wrapper,
        framework_wrapper,
        embedding_model,
        os.path.join(experiment_storage, f"persona_{i}"),
        svo_angle=svo_angle,
        svo_type=svo_type,
        disinfo=False,
    )

  # Initialize regular personas
  personas = {**leader_candidates}
  num_leaders = len(leader_types)
  assert 2 * num_leaders <= cfg.env.num_agents, (
      "Not enough personas for an election.")
  for i in range(num_leaders, cfg.env.num_agents):
    personas[f"persona_{i}"] = FishingPersona(
        cfg.agent,
        wrapper,
        framework_wrapper,
        embedding_model,
        os.path.join(experiment_storage, f"persona_{i}"),
    )

  # Initialize identities
  num_personas = cfg.personas.num
  identities = {
      f"persona_{i}": PersonaIdentity(
          agent_id=f"persona_{i}", **cfg.personas[f"persona_{i}"]
      )
      for i in range(num_personas)
  }
  # Log the identities to the consolidated log file.
  log_to_file(
      "persona_identities",
      {
          pid: {
              "name": (
                  identities[pid].name
                  if hasattr(identities[pid], "name")
                  else pid
              ),
              "type": (
                  str(leader_types[i])
                  if i < num_leaders
                  else str(SVOPersonaType.NONE)
              ),
              "config": {k: str(v) for k, v in persona_config.items()},
          }
          for i, (pid, persona_config) in enumerate([
              (
                  f"persona_{i}",
                  cfg.personas.get(
                      f"persona_{i}", cfg.personas.get("default", {})
                  ),
              )
              for i in range(cfg.env.num_agents)
          ])
      },
  )

  # Build mappings: agent_name_to_id maps from a candidate's name to its
  # internal id; agent_id_to_name reverses that mapping.
  agent_name_to_id = {obj.name: k for k, obj in identities.items()}
  agent_name_to_id["framework"] = "framework"
  agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

  # Initialize each persona and add references
  for persona in personas:
    personas[persona].init_persona(
        persona, identities[persona], social_graph=None
    )
  for persona in personas:
    for other_persona in personas:
      personas[persona].add_reference_to_other_persona(personas[other_persona])

  env_class = (
      FishingPerturbationEnv
      if cfg.env.class_name == "fishing_perturbation_env"
      else FishingConcurrentEnv
  )
  env = env_class(
      cfg.env,
      experiment_storage,
      agent_id_to_name,
      num_agents=cfg.env.num_agents)
  # Initialize the environment.
  sustainability_threshold = cfg.env.initial_resource_in_pool // (
      2 * cfg.env.num_agents
  )
  agent_id, obs = env.reset(
      sustainability_threshold=sustainability_threshold
  )
  curr_round = env.num_round

  # Run the first election
  current_time = datetime.datetime.now()
  winner, votes, leader_agendas = perform_election(
      personas,
      leader_candidates,
      current_time,
      wrapper,
      agent_id_to_name=agent_id_to_name,
      agent_name_to_id=agent_name_to_id,
  )
  election_results[0] = {
      "round": 0,
      "winner": winner,
      "agendas": leader_agendas,
      "votes": votes,
      "harvest_stats": None,
      "num_resources": env.internal_global_state["resource_in_pool"],
  }
  print(f"\nRound {curr_round} Election Winner: {winner}")
  log_to_file("election", election_results[0])
  logger.log_game({
      "round": 0,
      "election_winner": winner,
      "election_leader_agendas": leader_agendas,
      "election_votes": votes,
      "harvest_stats": None,
      "num_resources": env.internal_global_state["resource_in_pool"],
  })
  agenda = leader_agendas[winner]

  # Track harvest stats for each round.
  round_harvest_stats = collections.defaultdict(
      lambda: collections.defaultdict(int)
  )
  leader_harvest_report = None

  # MAIN SIM LOOP.
  while True:
    agent = personas[agent_id]
    # Set the current agenda and report.
    agent.update_agenda(agenda)
    agent.update_harvest_report(leader_harvest_report)
    action = agent.loop(obs)
    agent_id, obs, _, termination = env.step(action)

    # Check for harvest actions.
    if isinstance(action, PersonaActionHarvesting):
      round_harvest_stats[curr_round][agent.identity.name] = action.quantity

    stats = {}
    if hasattr(action, "stats"):
      for s in [
          "conversation_resource_limit",
          *[
              f"persona_{i}_collected_resource"
              for i in range(cfg.env.num_agents)
          ],
      ]:
        if s in action.stats:
          stats[s] = action.stats[s]
    logger.log_game({"num_resource": obs.current_resource_num, **stats})

    if np.any(list(termination.values())):
      logger.log_game({"num_resource": obs.current_resource_num}, last_log=True)
      break

    # Trigger another election?
    if curr_round != env.num_round:
      curr_round = env.num_round
      current_time = datetime.datetime.now()
      # TODO(rfaulk): Add disinfo to the report.
      if curr_round-1 in round_harvest_stats:
        leader_harvest_report = leaders_lib.make_harvest_report(
            personas, round_harvest_stats[curr_round-1])
      print(f"ROUND {curr_round} HARVEST REPORT:\n{leader_harvest_report}")
      # Update the harvest report for the leader.
      winner, votes, leader_agendas = perform_election(
          personas,
          leader_candidates,
          current_time,
          wrapper,
          agent_id_to_name=agent_id_to_name,
          agent_name_to_id=agent_name_to_id,
          # current_location=obs.current_location,
          last_winning_agenda=agenda,
          # TODO(rfaulk): When leaders can inject their own reports nodify this.
          harvest_report=leader_harvest_report,
          # harvest_stats=round_harvest_stats[curr_round-1],
      )
      agenda = leader_agendas[winner]
      election_results[curr_round] = {
          "round": curr_round,
          "winner": winner,
          "agendas": leader_agendas,
          "votes": votes,
          # Use the harvest stats from the last round.
          "harvest_stats": round_harvest_stats[curr_round-1],
          "num_resources": env.internal_global_state["resource_in_pool"],
      }
      print(f"\nRound {curr_round} Election Winner: {winner}")
      log_to_file("election", election_results[curr_round])
      logger.log_game({
          "round": curr_round,
          "election_winner": winner,
          "election_leader_agendas": leader_agendas,
          "election_votes": votes,
          # Use the harvest stats from the last round.
          "harvest_stats": round_harvest_stats[curr_round-1],
          "num_resources": env.internal_global_state["resource_in_pool"],
      })

  # Final harvest report.
  leader_harvest_report = leaders_lib.make_harvest_report(
            personas, round_harvest_stats[curr_round])
  print(f"\FINAL HARVEST REPORT - ROUND {curr_round}:\n{leader_harvest_report}")

  log_to_file("harvest", round_harvest_stats)
  log_to_file("sim-end", None)
  # if round_harvest_stats:
  #   logger.log_game(round_harvest_stats)
  env.save_log()
  for persona in personas:
    personas[persona].memory.save()

