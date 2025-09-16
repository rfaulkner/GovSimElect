"""Runs an election."""

import datetime
import json
import os

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from simulation.persona import EmbeddingModel
from simulation.persona import PersonaAgent
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

# from .agents.persona_leader import LeaderPersona
from .agents.persona_leader_clear import LeaderPersonaClear
from .agents.persona_leader_clear_noreasoning import LeaderPersonaClearNoReasoning
from .agents.persona_leader_gobbled import LeaderPersonaGobbled
from .agents.persona_leader_gobbled_reasoning import LeaderPersonaGobbledReasoning
from .agents.persona_v3 import FishingPersona
from .agents.persona_v3.cognition import utils as cognition_utils
from .agents.persona_v3.cognition.leader_agendas import prompt_leader_agenda_clear_direct
from .agents.persona_v3.cognition.leader_agendas import prompt_leader_agenda_clear_explain
from .agents.persona_v3.cognition.leader_agendas import prompt_leader_agenda_verbose_direct
from .agents.persona_v3.cognition.leader_agendas import prompt_leader_agenda_verbose_explain
from .environment import FishingConcurrentEnv
from .environment import FishingPerturbationEnv


cognition_utils.SYS_VERSION = "v3"

NUM_LEADERS = 4
NUM_VOTERS = 8
TOTAL_NUM_PERSONAS = NUM_LEADERS + NUM_VOTERS


def perform_election(
    personas: dict[str, PersonaAgent],
    leader_candidates: dict[str, PersonaAgent],
    current_time: str,
    wrapper: ModelWandbWrapper,
    agent_id_to_name: dict[int, str],
):
  """Runs an election among the leaders."""
  leader_agendas = {}
  # Get updated leader agendas using the leader prompt functions
  for pid in leader_candidates:
    if isinstance(leader_candidates[pid], LeaderPersonaClear):
      agenda, _ = prompt_leader_agenda_clear_explain(
          wrapper, leader_candidates[pid]
      )
    elif isinstance(leader_candidates[pid], LeaderPersonaGobbled):
      agenda, _ = prompt_leader_agenda_verbose_direct(
          wrapper, leader_candidates[pid]
      )
    elif isinstance(leader_candidates[pid], LeaderPersonaClearNoReasoning):
      agenda, _ = prompt_leader_agenda_clear_direct(
          wrapper, leader_candidates[pid]
      )
    elif isinstance(leader_candidates[pid], LeaderPersonaGobbledReasoning):
      agenda, _ = prompt_leader_agenda_verbose_explain(
          wrapper, leader_candidates[pid]
      )
    else:
      raise ValueError(
          f"Unknown leader persona type: {type(leader_candidates[pid])}"
      )
    leader_name = leader_candidates[pid].identity.name
    # print(f"\nAGENDA: {leader_candidates[pid].identity.name} {agenda}")
    leader_agendas[leader_name] = agenda

  # print(f"Leader Agendas:\n{leader_agendas}")
  votes = {}
  for persona_id in personas:
    # Only non-leader personas cast votes
    if persona_id not in leader_candidates:
      # # TODO(rfaulk): get memories.
      # focal_points = [current_context]
      # if len(current_conversation) > 0:
      #     # Last 4 utterances
      #     for _, utterance in current_conversation[-4:]:
      #         focal_points.append(utterance)
      # focal_points = personas[persona_id].retrieve.retrieve(
      #     focal_points, top_k=5
      # )
      current_location = "lake"  # or fishing_village?
      retireved_memory = personas[persona_id].retrieve.retrieve(
          [current_location], 10)
      vote, _ = personas[persona_id].act.participate_in_election(
          retireved_memory,  # retrieved memories; adjust as needed
          current_location,  # default location
          current_time,  # current time as string
          [leader.identity.name for _, leader in leader_candidates.items()],
          leader_agendas,
      )
      # print(f"Vote {personas[persona_id].identity.name}: {vote}")
      # Determine candidate identifier: if vote has attribute 'name', use it;
      # otherwise, use its string
      candidate_id = vote.name if hasattr(vote, "name") else str(vote)
      # Use the mapping to get the human-readable name; if not found, use
      # candidate_id as is.
      candidate_str = agent_id_to_name.get(candidate_id, candidate_id)
      votes[candidate_str] = votes.get(candidate_str, 0) + 1

  # Determine winner (as the candidate's human-readable name)
  winner = max(votes.items(), key=lambda x: x[1])[0]
  print("\nElection Voting Results:")
  for candidate, vote_count in votes.items():
    print(f"{candidate}: {vote_count} votes")
  print(f"\nWinner: {winner}")
  print("\nLeader Agendas:")
  for pid, agenda in leader_agendas.items():
    # Convert leader id to human-readable name using mapping
    print(f"\n{agent_id_to_name.get(pid, pid)}'s Agenda:")
    print(agenda)
  return winner, votes, leader_agendas


def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
  """Run the simulatiuon."""
  # Override experiment_storage to a new folder "multiturn_results" and create
  # it if it doesn't exist
  experiment_storage = "multiturn_results"
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
              OmegaConf.to_container(cfg)
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

  # Initialize leader candidates
  leader_candidates = {
      "persona_0": LeaderPersonaClear(
          cfg.agent,
          wrapper,
          embedding_model,
          os.path.join(experiment_storage, "persona_0"),
      ),
      "persona_1": LeaderPersonaGobbled(
          cfg.agent,
          wrapper,
          embedding_model,
          os.path.join(experiment_storage, "persona_1"),
      ),
      "persona_2": LeaderPersonaClearNoReasoning(
          cfg.agent,
          wrapper,
          embedding_model,
          os.path.join(experiment_storage, "persona_3"),
      ),
      "persona_3": LeaderPersonaGobbledReasoning(
          cfg.agent,
          wrapper,
          embedding_model,
          os.path.join(experiment_storage, "persona_4"),
      ),
  }

  # Initialize regular personas
  personas = {**leader_candidates}
  for i in range(NUM_LEADERS, TOTAL_NUM_PERSONAS):
    personas[f"persona_{i}"] = FishingPersona(
        cfg.agent,
        wrapper,
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
  leader_type_assignments = [
      "clear", "gobbled", "clear_noreasoning", "gobbled_reasoning"]
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
                  leader_type_assignments[i] if i < NUM_LEADERS else "follower"
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
              for i in range(TOTAL_NUM_PERSONAS)
          ])
      },
  )

  # Build mappings: agent_name_to_id maps from a candidate's name to its
  # internal id;
  # agent_id_to_name reverses that mapping.
  agent_name_to_id = {obj.name: k for k, obj in identities.items()}
  agent_name_to_id["framework"] = "framework"
  agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}
  print(f"Agent ID to Name:\n\n{agent_id_to_name}")

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
  env = env_class(cfg.env, experiment_storage, agent_id_to_name)
  agent_id, obs = env.reset()
  curr_round = env.num_round

  # Run the first election
  last_election_time = datetime.datetime.now()
  current_time_str = last_election_time.strftime("%H-%M-%S")
  winner, votes, leader_agendas = perform_election(
      personas, leader_candidates, current_time_str, wrapper, agent_id_to_name
  )
  election_results[0] = (winner, leader_agendas, votes)
  print(f"\nRound {curr_round} Election Winner: {winner}")
  log_to_file(f"election_{curr_round}", election_results[0])
  logger.log_game({
      "election_winner": winner,
      "election_leader_agendas": leader_agendas,
      "election_votes": votes,
  })
  agenda = leader_agendas[winner]

  # Main simulation loop
  while True:
    agent = personas[agent_id]
    # Set the current agenda.
    agent.update_agenda(agenda)
    action = agent.loop(obs)
    agent_id, obs, _, termination = env.step(action)

    stats = {}
    if hasattr(action, "stats"):
      for s in [
          "conversation_resource_limit",
          *[
              f"persona_{i}_collected_resource"
              for i in range(TOTAL_NUM_PERSONAS)
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
      current_time_str = current_time.strftime("%H-%M-%S")
      winner, votes, leader_agendas = perform_election(
          personas,
          leader_candidates,
          current_time_str,
          wrapper,
          agent_id_to_name,
      )
      agenda = leader_agendas[winner]
      election_results[curr_round] = (winner, leader_agendas, votes)
      print(f"\nRound {curr_round} Election Winner: {winner}")
      log_to_file("election", election_results[curr_round])
      logger.log_game({
          "round": curr_round,
          "election_winner": winner,
          "election_leader_agendas": leader_agendas,
          "election_votes": votes,
      })

  env.save_log()
  for persona in personas:
    personas[persona].memory.save()

