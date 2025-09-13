"""Compiles social analysis of simulation results.

This module implements a number of functions:

  read_persona_out: Reads the persona output file.
  read_env_out: Reads the environment output file.
  metric_degree_centrality: Computes the degree centrality metric.
"""

import json
import os
import re
import sys
from typing import Any

from collections import defaultdict
import networkx as nx
import pandas as pd


JSON_BASE_PATH = "/home/rfaulk/projects/aip-rgrosse/rfaulk/agent-ballot-box/multiturn_results"

PERSONA_FILE_LIST = [
    "persona_0/nodes.json",
    "persona_1/nodes.json",
    "persona_2/nodes.json",
    "persona_3/nodes.json",
    "persona_4/nodes.json",
]

ELECTIONS_DATA = "consolidated_results.json"
ENV_DATA = "log_env.json"
CHAT_TASK_STR= "Task: What would you say next in the group chat?"


def main(argv: list[str]):
  """Main example function."""

  if len(argv) > 1:
    sys.exit("Too many args.")
  print(f"args: {argv}")

  # Can probably remove.
  if len(argv) > 1:
    filename = argv[0]

  # Read the elections data and agent names.
  elections_data, agent_id_to_name = read_elections_data()
  # print(f"Agent ID to Name:\n\n{agent_id_to_name}")

  # Extract the agent network and stats.
  agent_network, env_out_dict = read_env_data(agent_id_to_name)
  # print(agent_network)
  # graph = nx.from_dict_of_dicts(agent_network)
  # print(graph)
  degree_centrality = metric_degree_centrality(agent_network)
  print(f"Degree Centrality:\n{degree_centrality}\n")
  betweeness_centrality = metric_betweeness_centrality(agent_network)
  print(f"Degree Betweeness:\n{betweeness_centrality}\n")
  importance_centrality = metric_importance_centrality(agent_network)
  print(importance_centrality)
  # Extract all persona names.
  # Extract all personas.

  # Read the persona output file.
  # persona_responses = {}
  # for filename in PERSONA_FILE_LIST:
  #   persona_responses[filename] = get_persona_responses(
  #       filename=PERSONA_FILE_LIST[0], agent_ids=agent_id_to_name)
  # print(f"Persona responses: {persona_responses}")


def get_persona_responses(
    filename: str,
    agent_ids: dict[str, str]) -> dict[str, list[str]]:
  """Read the persona output file."""
  # Load sample data from JSON file.
  text_list = []
  id_list = []
  round_list = []
  agent_name = list(agent_ids.values())
  with open(os.path.join(JSON_BASE_PATH, filename), "r") as f:
    json_dict = json.load(f)

  uid = 0
  round_id = 0
  user_name = ""
  for element in json_dict:
    if int(element["id"]) == 2:
      # Retrieve the user name from the description.
      user_name = element["description"].split()[0]
    if element["type"] == "CHAT":
      for utternace in element["conversation"]:
        if utternace[0] == user_name:
          # Only add the user's own utterances.
          text_list.append(utternace[1])
          if any(name in utternace[1] for name in agent_name):
            print(f"Found agent name in utterance: {utternace[1]}")
          id_list.append(f"user_{uid}")
          round_list.append(round_id)
          uid += 1
    elif element["type"] == "EVENT":
      if "Before everyone" in str(element["description"]):
        # Look for specific text heralding a new round.
        round_id += 1
    elif element["type"] == "THOUGHT":
      # Add all thoughts for this user.
      description = " ".join(element["description"].split("\n"))
      text_list.append(description)
      round_list.append(round_id)
      id_list.append(f"user_{uid}")
    uid += 1
  sample_data = {"text": text_list, "id": id_list, "round": round_list}
  # df = pd.DataFrame(sample_data)
  # print(f"Processing {len(df)} texts using batch processor...")
  return sample_data


def read_elections_data() -> tuple[list[dict[str, Any]], dict[str, str]]:
  """Reads the elections data & agent names."""
  with open(os.path.join(JSON_BASE_PATH, ELECTIONS_DATA), "r") as file:
    elections_data = [json.loads(line.rstrip()) for line in file]
  # extract agent names
  agent_id_to_name = {}
  for element in elections_data:
    if element["type"] == "persona_identities":
      for persona_id, agent_data in element["data"].items():
        agent_id_to_name[persona_id] = agent_data["name"]
      # extract election data - type 'election_0'
    elif element["type"] == "election":
      round_id = int(element["data"]["round"])
      winner_id = element["data"]["election_winner"]
      agendas = element["data"]["election_leader_agendas"]
      votes = element["data"]["election_votes"]
      # TODO: extract voting trends from this data.
  return elections_data, agent_id_to_name


def get_next_speaker_from_interaction(interaction: str) -> str:
  """Extracts the speaker from the interaction.
  
  Args:
    interaction: the agent's conversation response. 
  
  Returns:
    The next speaker they have chosen.
  """
  result = re.search(r"Next speaker: [a-zA-Z]+", interaction)
  if result:
    speaker = result.group(0)
    speaker = speaker.split()[-1]
    speaker = re.sub(r"[^a-zA-Z]+", "", speaker)
    return speaker
  else:
    return ""


def get_agent_references(
    interaction: str, 
    agents: list[str],
    speaker: str) -> list[str]:
  """Extracts references of other agents from the interaction.
  
  Args:
    interaction: the agent's conversation response. 
    agents: list of all agent names.
    speaker: the agent's own name.
  
  Returns:
    The next speaker they have chosen.
  """

  # Split on response. Return if there is none (not conversation).
  if CHAT_TASK_STR not in interaction:
    return []
  interaction = interaction.split(CHAT_TASK_STR)[-1]
  names = "|".join(agents)
  result = re.findall(rf"{names}", interaction)
  # Extract references not to oneself.
  refs = [ref for ref in result if ref != speaker]
  return refs


def read_env_data(agent_id_to_name: dict[str, str]):
  """Reads the env log and extracts the agent network.
  
  Args: 
    agent_id_to_name: mapping of agent IDs to names.
  
  Returns:
    The agent network DAG  and env out stats.
  
  Note that the ENV logs contain the following keys:
    
    ['agent_id', 'round', 'action', 'resource_in_pool_before_harvesting', 
     'resource_in_pool_after_harvesting', 'concurrent_harvesting',
     'resource_collected', 'wanted_resource', 'html_interactions',
     'agent_name', 'resource_limit', 'utterance'])
  """
  # Load sample data from JSON file.
  env_out_dict = defaultdict(lambda: defaultdict(list))

  # Initialise agent network.
  network_weights = {}
  for name_from in agent_id_to_name.values():
    network_weights[name_from] = {}
    for name_to in agent_id_to_name.values():
      network_weights[name_from][name_to] = {"weight": 0}

  with open(os.path.join(JSON_BASE_PATH, ENV_DATA), "r") as f:
    json_dict = json.load(f)
  for element in json_dict:
    if element["agent_id"] == "framework":
      continue
    agent_name = agent_id_to_name[element["agent_id"]]
    round = int(element["round"])

    # TODO(rfaulk): log these also.
    # action = element['action']
    # before_harvest = int(element['resource_in_pool_before_harvesting'])
    # after_harvest = int(element['resource_in_pool_after_harvesting'])
    # wanted_resource = int(element['wanted_resource'])

    interactions = element['html_interactions']
    if isinstance(interactions, list):
      for interaction in element['html_interactions']:
        env_out_dict[agent_name][round].append(interaction)
        next_speaker = get_next_speaker_from_interaction(interaction)
        agent_refs = get_agent_references(
            interaction,
            agents=[agent for agent in agent_id_to_name.values()],
            speaker=agent_name)
        # Add network weight.
        if next_speaker:
          network_weights[agent_name][next_speaker]["weight"] += 1
        for ref in agent_refs:
          network_weights[agent_name][ref]["weight"] += 1
    else:
      env_out_dict[agent_name][round].append(interactions)
      next_speaker = get_next_speaker_from_interaction(interactions)
      agent_refs = get_agent_references(
          interactions,
          agents=[agent for agent in agent_id_to_name.values()],
          speaker=agent_name)
      # Add network weight.
      if next_speaker:
        network_weights[agent_name][next_speaker]["weight"] += 1
      for ref in agent_refs:
        network_weights[agent_name][ref]["weight"] += 1
  return network_weights, env_out_dict


def metric_degree_centrality(
    agent_network: dict[str, dict[str, int]],
) -> dict[str, float]:
  """Compute degree centrality from network.
  
  Args:
    agent_network: the agent network DAG.
  
  Returns:
    A mapping of agent names to their degree centrality.

  This metric quantifies the number of direct connections an agent has, 
  indicating their prominence in communication and potential influence. 
  """
  degree_centrality = defaultdict(int)
  for agent_from, agent_edges in agent_network.items():
    for agent_to, weight in agent_edges.items():
      # Count all connections regardless of direction.
      degree_centrality[agent_from] += weight["weight"]
      degree_centrality[agent_to] += agent_network[agent_to][agent_from][
          "weight"
      ]
  return degree_centrality


def metric_betweeness_centrality(
    agent_network: dict[str, dict[str, int]]
) -> dict[str, float]:
  """Compute degree centrality from network.

  Args:
    agent_network: the agent network DAG.
  
  Returns:
    A mapping of agent names to their betweeness centrality.

  This identifies agents that act as bridges within the network by measuring
  their presence on the shortest paths between others, highlighting their role
  in facilitating communication and resource flow.
  """
  betweeness_centrality = defaultdict(int)
  G = nx.path_graph(agent_network)
  path = dict(nx.all_pairs_shortest_path(G))
  for agent_from, agent_edges in agent_network.items():
    for agent_to, _ in agent_edges.items():
      if agent_from == agent_to:
        continue
      for agent_interconnect in path[agent_from][agent_to][1:-2]:
        betweeness_centrality[agent_interconnect] += 1
  return betweeness_centrality


def metric_importance_centrality(agent_network: dict[str, dict[str, int]]):
  """Compute degree centrality from network.
  
    Args:
    agent_network: the agent network DAG.
  
  Returns:
    A mapping of agent names to their importance centrality.

  This assesses an agent’s influence based on the importance of their
  connections, identifying agents that hold significant sway through their
  associations.
  """
  G = nx.path_graph(agent_network)
  importance_centrality = nx.eigenvector_centrality(G)
  return importance_centrality


if __name__ == "__main__":
  main(sys.argv[1:])

