"""Compiles social analysis of simulation results.

This module implements a number of functions:

  read_persona_out: Reads the persona output file.
  read_env_out: Reads the environment output file.
  metric_degree_centrality: Computes the degree centrality metric.
  metric_betweeness_centrality: Computes the betweeness centrality metric.
  metric_importance_centrality: Computes the importance centrality metric.
  election_metrics: Computes the election metrics.
  compute_cummulative_resource: Computes cummulative resource for each agent.
  metric_gini_coefficient: Computes the Gini coefficients across cycles.
  gini: Computes the Gini coefficient for a population.
"""

import collections
import json
import os
import re
import sys
from typing import Any

import networkx as nx
import numpy as np
# import pandas as pd

MAX_CYCLES = 12

JSON_BASE_PATH = ""
MODEL_PATH_LIST = []

MODEL_PATH_LIST_MISTRAL_7 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/mistralai/Mistral-7B-Instruct-v0.2_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/mistralai/Mistral-7B-Instruct-v0.2_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/mistralai/Mistral-7B-Instruct-v0.2_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/mistralai/Mistral-7B-Instruct-v0.2_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/mistralai/Mistral-7B-Instruct-v0.2_run_4",
]


MODEL_PATH_LIST_GEMMA_27 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemini-2.5-flash_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemini-2.5-flash_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemini-2.5-flash_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemini-2.5-flash_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemini-2.5-flash_run_4",
]

MODEL_PATH_LIST_GEMINI_2_5_FLASH = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemma-3-27b-it_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemma-3-27b-it_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemma-3-27b-it_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemma-3-27b-it_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-google/gemma-3-27b-it_run_4",
]

MODEL_PATH_LIST_LLAMA_8 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/meta-llama/Meta-Llama-3-8B-Instruct_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/meta-llama/Meta-Llama-3-8B-Instruct_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/meta-llama/Meta-Llama-3-8B-Instruct_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/meta-llama/Meta-Llama-3-8B-Instruct_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/meta-llama/Meta-Llama-3-8B-Instruct_run_4",
]

MODEL_PATH_LIST_LLAMA_70 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_elect_6",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_elect_7",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_elect_8",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_elect_9",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_elect_10",
]

MODEL_PATH_LIST_LLAMA_70_VANILLA = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_6",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_7",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_8",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_9",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-meta-llama/llama-3-70b-instruct_run_10",
]

MODEL_PATH_LIST_QWEN_72 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-72B-Chat-GPTQ-Int4_run_4",
]

MODEL_PATH_LIST_QWEN_110 = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4_run_4",
]

MODEL_PATH_LIST_GPT_4_turbo = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4-turbo-2024-04-09_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4-turbo-2024-04-09_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4-turbo-2024-04-09_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4-turbo-2024-04-09_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4-turbo-2024-04-09_run_4",
]

MODEL_PATH_LIST_GPT_3_5_turbo = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-3.5-turbo-0125_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-3.5-turbo-0125_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-3.5-turbo-0125_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-3.5-turbo-0125_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-3.5-turbo-0125_run_4",
]

MODEL_PATH_LIST_GPT_4o = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_0",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_1",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_2",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_3",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_4",
]

MODEL_PATH_LIST_GPT_4O_50_AGENTS = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_5",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_6",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_7",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_8",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/gpt/gpt-4o-2024-05-13_run_9",
]

MODEL_PATH_LIST_SONNET = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/sonnet_run_0/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/sonnet_run_1/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/sonnet_run_2/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/sonnet_run_3/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/sonnet_run_4/",
]

MODEL_PATH_LIST_HAIKU = [
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/haiku_run_0/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/haiku_run_1/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/haiku_run_2/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/haiku_run_3/",
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/openrouter-anthropic/haiku_run_4/",
]

PERSONA_FILE_LIST = [
    "persona_0/nodes.json",
    "persona_1/nodes.json",
    "persona_2/nodes.json",
    "persona_3/nodes.json",
    "persona_4/nodes.json",
]

ELECTIONS_DATA = "consolidated_results.json"
ENV_DATA = "log_env.json"
CHAT_TASK_STR = "Task: What would you say next in the group chat?"

DEBUG = True


def main(argv: list[str]):
  """Main example function."""

  if len(argv) > 1:
    sys.exit("Too many args.")

  totals_map = {
      "degree_centrality": collections.defaultdict(float),
      "edge_centrality": collections.defaultdict(float),
      "importance_centrality": collections.defaultdict(float),
      "gini_cycle_coefficients": [0.0] * 12,
      "survival_time": 0.0,
      "survived": 0.0,
      "harvest_by_agent": collections.defaultdict(float),
      "total_harvest": 0.0,
      "election_winners": collections.defaultdict(float),
      "election_total_votes": collections.defaultdict(float),
      "election_consecutive_wins": collections.defaultdict(float),
  }
  global JSON_BASE_PATH
  global MODEL_PATH_LIST
  MODEL_PATH_LIST = MODEL_PATH_LIST_SONNET

  for model_path in MODEL_PATH_LIST:

    print(f"Processing {model_path}...")
    JSON_BASE_PATH = model_path

    # Read the elections data and agent names.
    elections_data, agent_id_to_name, harvest_data = read_elections_data()
    print(f"Agent ID to Name: {agent_id_to_name}\n")
    
    # Extract the agent network and stats.
    agent_network, inverse_weight_network, _ = read_env_data(agent_id_to_name)
    if DEBUG:
      print(f"Agent Network: {agent_network}\n")
      print(f"Inverse Weight Network: {inverse_weight_network}\n")
      graph = nx.from_dict_of_dicts(agent_network)
      print(f"Agent network graph: {graph}")

    # Degree centrality.
    degree_centrality = metric_degree_centrality(agent_network)
    for agent_name, degree in degree_centrality.items():
      totals_map["degree_centrality"][agent_name] += degree
    if DEBUG:
      print(f"Degree centrality: {degree_centrality}\n")

    # Edge centrality.
    betweeness_centrality = metric_betweeness_centrality(inverse_weight_network)
    for agent_name, degree in betweeness_centrality.items():
      totals_map["edge_centrality"][agent_name] += degree
    if DEBUG:
      print(f"Betweeness centrality: {betweeness_centrality}\n")

    # Importance centrality.
    importance_centrality = {}
    try:
      importance_centrality = metric_importance_centrality(agent_network)
      for agent_name, degree in importance_centrality.items():
        totals_map["importance_centrality"][agent_name] += degree
    except Exception as e:
      print(f"Failed to compute importance centrality: {e}")
      pass
    if DEBUG:
      print(f"Importance centrality: {importance_centrality}\n")

    # Sustainability.
    survival_time = len(harvest_data)
    survived = survival_time == 12
    harvest_by_agent = collections.defaultdict(int)

    for _, data in harvest_data.items():
      for agent_name, resources in data.items():
        harvest_by_agent[agent_name] += int(resources)

    totals_map["survival_time"] += float(survival_time)
    totals_map["survived"] += float(survived)
    for agent_name, harvest in harvest_by_agent.items():
      totals_map["harvest_by_agent"][agent_name] += float(harvest)
    totals_map["total_harvest"] += float(sum(harvest_by_agent.values()))

    # Elections Metrics.
    winners, total_votes, consecutive_wins = election_metrics(elections_data)
    for agent_name, winner in winners.items():
      totals_map["election_winners"][agent_name] += float(winner)
    for agent_name, total in total_votes.items():
      totals_map["election_total_votes"][agent_name] += float(total)
    for agent_name, consecutive in consecutive_wins.items():
      totals_map["election_consecutive_wins"][agent_name] += float(consecutive)
    if DEBUG:
      print(f"Election winners: {winners}\n")
      print(f"Election total votes: {total_votes}\n")
      print(f"Election consecutive wins: {consecutive_wins}\n")

    # Measure Inequality va Gini.
    gini_cycle_coefficients, _ = metric_gini_coefficient(harvest_data)
    for idx, gini_coeff in enumerate(gini_cycle_coefficients):
      totals_map["gini_cycle_coefficients"][idx] += gini_coeff
    if DEBUG:
      print(f"Gini cycle coefficients: {gini_cycle_coefficients}\n")

  for totals_key, totals_value in totals_map.items():
    if isinstance(totals_value, collections.defaultdict):
      for key, value in totals_value.items():
        totals_map[totals_key][key] = value / len(MODEL_PATH_LIST)
    elif isinstance(totals_value, list):
      for idx, value in enumerate(totals_value):
        totals_map[totals_key][idx] = value / len(MODEL_PATH_LIST)
    else:
      totals_map[totals_key] = totals_value / len(MODEL_PATH_LIST)
  totals_map = dict(totals_map)
  print(f"Totals:\n{json.dumps(totals_map, indent=2)}")


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
  elections_dict = {}
  harvest_data_by_cycle = {}
  agent_id_to_name = {}
  for element in elections_data:
    if element["type"] == "persona_identities":
      for persona_id, agent_data in element["data"].items():
        agent_id_to_name[persona_id] = agent_data["name"]
      # extract election data - type 'election_0'
    elif element["type"] == "election":
      round_id = int(element["data"]["round"])
      elections_dict[round_id] = {
          "winner": element["data"]["winner"],
          # "agendas": element["data"]["agendas"],
          "votes": element["data"]["votes"],
          "harvest_stats": element["data"]["harvest_stats"],
      }
    elif element["type"] == "harvest":
      harvest_data_by_cycle = element["data"]

  return elections_dict, agent_id_to_name, harvest_data_by_cycle


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


def add_reference(
    network: dict[str, dict[str, dict[str, float]]],
    from_agent: str,
    to_agent: str,
    weight: float = 1.0,
) -> dict[str, dict[str, dict[str, float]]]:
  """Adds or updates a reference to the network by a weight."""
  if from_agent not in network:
    network[from_agent] = {}
  if to_agent not in network[from_agent]:
    network[from_agent][to_agent] = {"weight": weight}
  else:
    network[from_agent][to_agent]["weight"] += weight
  return network


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
  env_out_dict = collections.defaultdict(lambda: collections.defaultdict(list))

  # Initialise agent network.
  network_weights = {}
  inverse_weight_network = {}

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
          add_reference(network_weights, agent_name, next_speaker)
        for ref in agent_refs:
          add_reference(network_weights, agent_name, ref)
    else:
      env_out_dict[agent_name][round].append(interactions)
      next_speaker = get_next_speaker_from_interaction(interactions)
      agent_refs = get_agent_references(
          interactions,
          agents=[agent for agent in agent_id_to_name.values()],
          speaker=agent_name)
      # Add network weight.
      if next_speaker:
        add_reference(network_weights, agent_name, next_speaker)
      for ref in agent_refs:
        add_reference(network_weights, agent_name, ref)

  # Make an inverse network.
  for name_from in network_weights:
    for name_to in network_weights[name_from]:
      add_reference(
          inverse_weight_network,
          name_from,
          name_to,
          1.0 / network_weights[name_from][name_to]["weight"],
      )
  return network_weights, inverse_weight_network, env_out_dict


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
  degree_centrality = collections.defaultdict(int)
  for agent_from, agent_edges in agent_network.items():
    for agent_to, weight in agent_edges.items():
      # Count all connections regardless of direction.
      degree_centrality[agent_from] += weight["weight"]
      degree_centrality[agent_to] += agent_network[agent_from][agent_to][
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
  betweeness_centrality = collections.defaultdict(int)
  graph = nx.from_dict_of_dicts(agent_network)
  path = dict(nx.all_pairs_shortest_path(graph))
  for agent_from, agent_edges in path.items():
    for agent_to, nodes in agent_edges.items():
      if agent_from == agent_to or len(nodes) < 3:
        continue
      for agent_interconnect in nodes[1:-1]:
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
  graph = nx.from_dict_of_dicts(agent_network)
  importance_centrality = nx.eigenvector_centrality(graph)
  return importance_centrality


def election_metrics(elections_data: dict[int, dict[str, Any]]):
  """Compute election metrics from elections data."""
  # TODO(rfaulk): Implement this.
  winners = collections.defaultdict(int)
  consecutive_wins = collections.defaultdict(int)
  total_votes = collections.defaultdict(int)
  for cycle, data in elections_data.items():
    winners[data["winner"]] += 1
    for agent_id, vote_tally in data["votes"].items():
      total_votes[agent_id] += vote_tally
    if cycle > 0:
      if (
          winners[data["winner"]]
          == winners[elections_data[cycle - 1]["winner"]]
      ):
        consecutive_wins[data["winner"]] += 1
  return winners, total_votes, consecutive_wins


def compute_cummulative_resource(
    agent_resources: dict[str, list[int]],
) -> tuple[dict[str, list[int]], dict[int, str]]:
  """Compute cumulative resource from network.

  Args:
    agent_resources: the agent resources.

  Returns:
    A numpy array of cumulative resources for each agent.
  
  e.g.
      {'A': [1, 2, 3], 'B': [4, 5, 6]}

  yields:
      np.array([[1, 3, 6], [4, 9, 15]])
  """
  cumulative_agent_resources = np.ndarray(
      shape=(len(agent_resources), MAX_CYCLES)
  )
  idx = 0
  agent_map = {}
  for agent_id, resources in agent_resources.items():
    agent_map[idx] = agent_id
    cumulative_resources = np.cumsum(resources)
    cumulative_agent_resources[idx, :] = cumulative_resources
    idx += 1
  return cumulative_agent_resources, agent_map


def gini(x):
  # (Warning: This is a concise implementation, but it is O(n**2)
  # in time and memory, where n = len(x).  *Don't* pass in huge
  # samples!)

  # Mean absolute difference
  mad = np.abs(np.subtract.outer(x, x)).mean()
  # Relative mean absolute difference
  rmad = mad / np.mean(x)
  # Gini coefficient
  g = 0.5 * rmad
  return g


def metric_gini_coefficient(
    agent_resources: dict[int, dict[str, int]]) -> list[float]:
  """Compute gini coefficient from network.

  Args:
    agent_resources: the agent resources.

  Returns:
    The gini coefficient of the network.

  This assesses the degree of inequality in the network, quantifying the
  disparity in influence across agents.
  """
  agent_resources_list = collections.defaultdict(list)
  # TODO(rfaulk): Compute the range limit.
  n_cycles_survived = max(int(cycle) for cycle in agent_resources) + 1
  for cycle in range(n_cycles_survived):
    for agent_name, harvest in agent_resources[str(cycle)].items():
      agent_resources_list[agent_name].append(harvest)
  for _ in range(n_cycles_survived, MAX_CYCLES):
    for agent_name in agent_resources_list:
      agent_resources_list[agent_name].append(0.0)
  cumulative_agent_resources, agent_map = compute_cummulative_resource(
      agent_resources_list
  )
  gini_cycle_coefficients = []
  for cycle in range(np.shape(cumulative_agent_resources)[1]):
    gini_cycle_coefficients.append(gini(cumulative_agent_resources[:, cycle]))
  return gini_cycle_coefficients, agent_map

if __name__ == "__main__":
  main(sys.argv[1:])

