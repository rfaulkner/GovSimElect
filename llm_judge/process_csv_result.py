"""Read the csv analysis results."""

import collections
import csv
import json
import math
import os
import statistics
import sys


BASE_PATH = (
    '/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/llm_judge/results'
)

ID_IDX = 1
CATEGORY_IDX = 4


file_list_coop = [
    'sentiments_one_prosocial_disinfo_True_cooperation.csv',
    'sentiments_one_competitive_disinfo_True_cooperation.csv',
    'sentiments_one_individualistic_disinfo_True_cooperation.csv',
    'sentiments_one_altruistic_disinfo_True_cooperation.csv',
    'sentiments_one_prosocial_disinfo_False_cooperation.csv',
    'sentiments_one_competitive_disinfo_False_cooperation.csv',
    'sentiments_one_individualistic_disinfo_False_cooperation.csv',
    'sentiments_one_altruistic_disinfo_False_cooperation.csv',
]

file_list_persuasion = [
    'sentiments_one_prosocial_disinfo_True_persuasion.csv',
    'sentiments_one_competitive_disinfo_True_persuasion.csv',
    'sentiments_one_individualistic_disinfo_True_persuasion.csv',
    'sentiments_one_altruistic_disinfo_True_persuasion.csv',
    'sentiments_one_prosocial_disinfo_False_persuasion.csv',
    'sentiments_one_competitive_disinfo_False_persuasion.csv',
    'sentiments_one_individualistic_disinfo_False_persuasion.csv',
    'sentiments_one_altruistic_disinfo_False_persuasion.csv',
]

COOPERATIVE_CATEGORIES = [
    'moral considerations',
    'cooperative argument',
    'social norms and conformity',
    'reputation concerns',
    'psychological factors'
]

PERSUATION_CATEGORIES = [
    'Logos',
    'Pathos',
    'Ethos',
    'Neutral'
]


def default_to_regular(d):
  if isinstance(d, collections.defaultdict):
    d = {k: default_to_regular(v) for k, v in d.items()}
  return d


def compute_coop_index(data) -> float:
  """Compute the cooperative index."""
  coop = 0
  not_coop = 0
  for cat, count in data.items():
    if cat in COOPERATIVE_CATEGORIES:
      coop += count
    else:
      not_coop += data[cat]
  return float(coop) / float(not_coop + coop)


def compute_persuasion_index(data) -> dict[str, float]:
  """Compute the persuasion index."""
  total = 0
  out_data = {'logos': 0.0, 'pathos': 0.0, 'ethos': 0.0, 'neutral': 0.0}
  for _, count in data.items():
    total += count
  for cat, count in data.items():
    out_data[cat] = float(count) / float(total)
  return out_data


def main(argv: list[str]) -> None:
  del argv
  all_results = {}
  for in_file in file_list_coop + file_list_persuasion:
    csv_dict = collections.defaultdict(lambda: collections.defaultdict(int))
    # Check if this file exists, otherwise skip.
    if not os.path.exists(os.path.join(BASE_PATH, in_file)):
      print(f'File {in_file} does not exist.')
      continue
    with open(os.path.join(BASE_PATH, in_file), 'r') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      line_count = 0
      for row in csv_reader:
        if line_count == 0:
          # Skip the header row.
          line_count += 1
        else:
          # There can be multiple sentiments per text.
          cycle = int(row[ID_IDX])
          sentiments = row[CATEGORY_IDX].split(',')
          for sentiment in sentiments:
            csv_dict[cycle][sentiment.strip().lower()] += 1
          line_count += 1
    sentiment_indices = []
    # Iterate over the seeds and compute the mean and standard error.
    result_key = in_file.split('.')[0]
    if 'cooperation' in in_file:
      # Handles cooperation taxonomy.
      for _, data in csv_dict.items():
        sentiment_indices.append(compute_coop_index(data))
      mean_value = statistics.mean(sentiment_indices)
      serror = statistics.stdev(sentiment_indices) / math.sqrt(
          len(sentiment_indices))
      all_results[result_key] = [mean_value, serror]
    elif 'persuasion' in in_file:
      # Handles persuasion taxonomy.
      all_persuasion_indices = {
          'logos': [],
          'pathos': [],
          'ethos': [],
          'neutral': []
      }
      final_persuasion_indices = collections.defaultdict(list)
      for _, data in csv_dict.items():
        persuasion_indicies = compute_persuasion_index(data)
        for key, val in persuasion_indicies.items():
          t_key = 'neutral' if key == 'other' else key
          all_persuasion_indices[t_key].append(val)
      print(f'all_persuasion_indices: {all_persuasion_indices}')
      for key, val in all_persuasion_indices.items():
        if not val:
          final_persuasion_indices[key] = [0.0, 0.0]
        elif len(val) == 1:
          final_persuasion_indices[key] = [val[0], 0.0]
        else:
          mean_value = statistics.mean(val)
          serror = statistics.stdev(val) / math.sqrt(len(val))
          final_persuasion_indices[key] = [mean_value, serror]
      all_results[result_key] = final_persuasion_indices

  # Write all results to a single file.
  fname = os.path.join(BASE_PATH, 'sentiments_merged.json')
  print(f'Writing to "{fname}" ...')
  with open(fname, 'w+') as f:
    json.dump(all_results, f)


if __name__ == '__main__':
  main(sys.argv[1:])

