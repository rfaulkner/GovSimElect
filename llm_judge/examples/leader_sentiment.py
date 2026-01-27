#!/usr/bin/env python3

"""Basic usage example for the LLM Judge package.

This example shows how to:
1. Set up an API client
2. Load a taxonomy
3. Create an LLM judge
4. Classify individual texts
5. Process multiple texts in batch
"""

# pylint: disable=g-import-not-at-top

import enum
import json
import os
import sys
from typing import Any

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api_clients import AzureOpenAIClient
from api_clients import DeepSeekClient
from api_clients import OpenRouterClient
from judge import LLMJudge
import pandas as pd
from processors import BatchProcessor
from taxonomy import Taxonomy


# SVO Persona Types.
class LeaderPopulationType(enum.Enum):
  BALANCED = "balanced"
  LEAN_ALTRUISTIC = "lean_altruistic"
  LEAN_COMPETITIVE = "lean_competitive"
  ONE_COMPETITIVE = "one_competitive"
  ONE_ALTRUISTIC = "one_altruistic"
  ONE_PROSOCIAL = "one_prosocial"
  ONE_INDIVIDUALISTIC = "one_individualistic"
  NONE = "none"


N_SEEDS = 10
N_CYCLES = 6

IN_BASE_PATH = "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0"
IN_MODEL = "gpt"

JSON_BASE_PATH = "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/llm_judge/"
DEFAULT_OUTFILE = "sentiments"
FIRST_USER = "Julia"


class TaxonomyType(enum.Enum):
  """Taxonomy for sentiment analysis."""
  SVO = "svo"
  PERSUASION = "persuasion"
  COOPERATION = "cooperation"

POPULATION = LeaderPopulationType.ONE_PROSOCIAL
DISINFO = False
TAXONOMY_TYPE = TaxonomyType.COOPERATION


def svo_taxonomy():
  """Create a custom taxonomy for email classification."""

  # Create empty taxonomy
  taxonomy = Taxonomy()

  taxonomy.add_category(
      "None",
      "No particular Social Value Orientation (SVO) Type:.",
  )

  taxonomy.add_category(
      "Altrusitic",
      "Social Value Orientation (SVO) Type: An altruistic individual is"
      " motivated to help others and will sacrifice their own outcomes to"
      " benefit someone else, showing low self-interest.",
  )

  taxonomy.add_category(
      "Competitive",
      "Social Value Orientation (SVO) Type: A competitive individual strives to"
      " maximize their own outcomes and, in addition, seeks to minimize the"
      " other person's outcomes, finding satisfaction in doing better than"
      " others (winning).",
  )

  taxonomy.add_category(
      "Individualistic",
      "Social Value Orientation (SVO) Type: An individualistic person is"
      " concerned only with their own outcomes and is largely indifferent to"
      " the outcomes of others (doing well for oneself).",
  )

  taxonomy.add_category(
      "Prosocial",
      "Social Value Orientation (SVO) Type: A prosocial individual (often"
      " grouped with cooperative types) aims to maximize both their own and"
      " others' outcomes (doing well together) or ensure fairness and equality"
      " in outcomes.",
  )

  return taxonomy


def persuasion_taxonomy():
  """Create a custom taxonomy for email classification."""

  # Create empty taxonomy
  taxonomy = Taxonomy()

  # Add categories for email classification
  taxonomy.add_category(
      "Neutral",
      "The statement made by the speaker is a neutral phrase not aiming to"
      " persuade, convince, or influence.",
  )

  taxonomy.add_category(
      "Logos",
      "A statement meant to persuade through logical reasoning rather than"
      " through emotions or appeal to authority.",
  )

  taxonomy.add_category(
      "Pathos",
      "A statement meant to persuade through appeal to the listener's emotions"
      " rather than through logical reasoning or authority.",
  )

  taxonomy.add_category(
      "Ethos",
      "A statement meant to persuade through appeal to the listener's respect"
      " for authority and norms rather than through logical reasoning or"
      " emotion.",
  )

  return taxonomy


def setup_api_client():
  """Set up your API client. Choose one of the following."""

  # Option 1: Azure OpenAI
  if os.getenv("AZURE_API_KEY"):
    client = AzureOpenAIClient(
        api_key=os.getenv("AZURE_API_KEY"),
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_version="2024-12-01-preview",
    )
    return client

  # Option 2: OpenRouter
  elif os.getenv("OPENROUTER_API_KEY"):
    client = OpenRouterClient(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini"),
    )
    return client

  # Option 3: DeepSeek
  elif os.getenv("DEEPSEEK_API_KEY"):
    client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"), model_name="deepseek-chat"
    )
    return client

  else:
    print("No API credentials found in environment variables.")
    print("Please set one of the following:")
    print("- AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME")
    print("- OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME")
    print("- DEEPSEEK_API_KEY")
    sys.exit(1)


def read_json() -> list[dict[str, Any]]:
  """Read a JSON file and return a list of events."""
  events_list = []
  in_file_path_prefix = (
      f"gpt-4o-2024-05-13_disinfo_{DISINFO}_population_{POPULATION.value}"
  )
  for i in range(1, N_SEEDS + 1):
    in_file = os.path.join(
        IN_BASE_PATH,
        IN_MODEL,
        in_file_path_prefix + f"_run_{i}/persona_0/nodes.json",
    )
    print(f"Reading {in_file}...")
    with open(in_file, "r") as f:
      events = json.load(f)
      for i, _ in enumerate(events):
        events[i]["seed"] = i-1
      events_list += events
  return events_list


def read_response_json() -> list[dict[str, Any]]:
  """Read a JSON file and return a list of responses."""
  responses_list = []
  in_file_path_prefix = (
      f"gpt-4o-2024-05-13_disinfo_{DISINFO}_population_{POPULATION.value}"
  )
  for i in range(1, N_SEEDS + 1):
    in_file = os.path.join(
        IN_BASE_PATH,
        IN_MODEL,
        in_file_path_prefix + f"_run_{i}/responses.json",
    )
    print(f"Reading {in_file}...")
    with open(in_file, "r") as f:
      for line in f:
        data = json.loads(line)
        data["seed"] = i-1
        responses_list += [data]
  return responses_list


def parse_nodes():
  """Parse the nodes JSON file."""
  text_list = []
  id_list = []
  round_id = 0
  user_name = ""
  events_list = read_json()

  for element in events_list:
    seed = element["seed"]
    utt_id = N_CYCLES * seed + round_id
    if int(element["id"]) == 2:
      # Retrieve the user name from the description.
      user_name = element["description"].split()[0]
    if element["type"] == "CHAT":
      for utternace in element["conversation"]:
        if utternace[0] == user_name:
          # Only add the user's own utterances.
          text_list.append(utternace[1])
          id_list.append(utt_id)
    elif element["type"] == "EVENT":
      if "Before everyone" in str(element["description"]):
        # Look for specific text heralding a new round.
        round_id += 1
    elif element["type"] == "THOUGHT":
      # Add all thoughts for this user.
      description = " ".join(element["description"].split("\n"))
      text_list.append(description)
      id_list.append(utt_id)
  sample_data = {"text": text_list, "id": id_list}
  return sample_data


def parse_responses(include_actions: bool = True) -> dict[str, list[str]]:
  """Parse the responses JSON file.

  Args:
    include_actions: Whether to include action responses (True) or only
      converse responses (False).
  
  Response types are:
    - action_response: the agent's response to a specific action.
    - converse_response: the agent's response to a user utterance.

  Returns:
    A dict of filtered responses.
  """
  action_responses = []
  converse_responses = []
  action_ids = []
  converse_ids = []
  responses_list = read_response_json()
  for event in responses_list:
    seed = event["seed"]
    data = event["data"]
    response_type = event["type"]
    if data["speaker"] == FIRST_USER:
      if response_type == "action_response" and include_actions:
        action_responses.append(data["reasoning"])
        action_ids.append(seed)
      elif response_type == "converse_response":
        converse_responses.append(data["utterance"])
        converse_ids.append(seed)
      else:
        continue
  sample_data = {
      "text": action_responses + converse_responses,
      "id": action_ids + converse_ids,
      }
  return sample_data


def main(argv: list[str]):
  """Main example function."""

  print(f"args: {argv}")

  global POPULATION
  global DISINFO
  global TAXONOMY_TYPE

  if len(argv):
    POPULATION = LeaderPopulationType(argv[0])
  if len(argv) > 1:
    DISINFO = argv[1] == "true"
  if len(argv) > 2:
    TAXONOMY_TYPE = TaxonomyType(argv[2])

  # Set output file.
  results_path = os.path.join(JSON_BASE_PATH, "results")
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  out_file = os.path.join(
      results_path,
      f"{DEFAULT_OUTFILE}_{POPULATION.value}_disinfo_{DISINFO}_{TAXONOMY_TYPE.value}.csv",
  )

  print("=== LLM Judge Basic Usage Example ===\n")

  # 1. Set up API client
  print("1. Setting up API client...")
  api_client = setup_api_client()

  # Verify connection
  success, message = api_client.verify_connection()
  if not success:
    print(f"API connection failed: {message}")
    return
  print(f"✓ API connection successful: {message}\n")

  # 2. Load taxonomy
  print(f"2. Loading taxonomy for {TAXONOMY_TYPE.value} ..")
  if TAXONOMY_TYPE == TaxonomyType.PERSUASION:
    taxonomy = persuasion_taxonomy()
  elif TAXONOMY_TYPE == TaxonomyType.SVO:
    taxonomy = svo_taxonomy()
  elif TAXONOMY_TYPE == TaxonomyType.COOPERATION:
    taxonomy_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "cooperation_taxonomy.json"
    )
    taxonomy = Taxonomy.from_json_file(taxonomy_path)
  else:
    raise ValueError(f"Unknown taxonomy type: {TAXONOMY_TYPE}")
  print(f"✓ Loaded taxonomy with {len(taxonomy)} categories")
  print("Categories:", ", ".join(taxonomy.get_categories()))
  print()

  # 3. Create LLM judge
  print("3. Creating LLM judge...")
  judge = LLMJudge(api_client, taxonomy, temperature=0)
  print("✓ LLM judge created\n")

  # 5. Batch processing example
  print("5. Batch processing GovSim data:")

  # Load sample data from JSON file.
  # sample_data = parse_nodes()
  sample_data = parse_responses(TAXONOMY_TYPE == TaxonomyType.COOPERATION)

  # Combine these data somehow.
  df = pd.DataFrame(sample_data)
  print(f"Processing {len(df)} texts using batch processor...")

  # Create batch processor
  batch_processor = BatchProcessor(judge, max_workers=10, batch_size=3)

  # Process the dataframe
  df_with_results = batch_processor.process_dataframe(
      df, text_column="text", metadata_columns=["id", "round"]
  )

  # Display results
  print("\nBatch processing results:")
  for _, row in df_with_results.iterrows():
    print(
        f"ID {row['id']}: "
        f"{row['classification_justification']} "
        f"(confidence: {row['classification_confidence']:.2f})"
    )

  # Save results
  if os.path.exists(f"{out_file}"):
    os.remove(out_file)
    print(f"Removed old '{out_file}' ...")
  df_with_results.to_csv(out_file, index=False)
  print(f"\n✓ Results saved to {out_file}")

  # Show API usage statistics
  print("\nAPI Statistics:")
  print(f"Total cost: ${api_client.get_total_cost():.4f}")
  print(f"Success rate: {api_client.get_success_rate():.1f}%")

  print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
  main(sys.argv[1:])

