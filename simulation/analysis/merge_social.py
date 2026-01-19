"""Merge the social results for a model."""

from collections.abc import Sequence
import enum
import json
import sys

POPULATIONS = (
    "balanced",
    "lean_altruistic",
    "lean_competitive",
    "one_prosocial",
    "one_competitive",
    "one_individualistic",
    "one_altruistic",
    "none",
)


class ModelPaths(enum.Enum):
  QWEN_72B = "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4"
  QWEN_110B = "Qwen/Qwen1.5-110B-Chat-GPTQ-Int4"
  GPT_4O = "gpt/gpt-4o-2024-05-13"
  GPT_4_1 = "gpt/gpt-4.1-2025-04-14"
  GEMINI_2_5_FLASH = "openrouter-google/gemini-2.5-flash"

BASE_PATH = "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0"
MODEL_TYPE = ModelPaths.GPT_4_1


def main(argv: Sequence[str]) -> None:

  if len(argv) > 1:
    raise ValueError("Too many command-line arguments.")

  # compose paths
  results = {}
  model_family, model_name = MODEL_TYPE.value.split("/")
  for population in POPULATIONS:
    for disinfo in [True, False]:
      fname = f"{BASE_PATH}/{model_family}/analysis/{model_name}_disinfo_{disinfo}_population_{population}.json"
      print(f"Reading {fname}...")
      with open(fname, "r") as f:
        key = f"population_{population}_disinfo_{disinfo}"
        results[key] = json.load(f)

  fname = f"{BASE_PATH}/{model_family}/analysis/{model_name}_merged.json"
  print(f"Writing {fname}...")
  with open(fname, "w+") as f:
    json.dump(results, f)

if __name__ == "__main__":
  main(sys.argv[1:])

