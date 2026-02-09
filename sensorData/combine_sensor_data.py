#!/usr/bin/env python3
# Abtin Olaee 2025

import json
from pathlib import Path
from tqdm import tqdm 

sensor_data_directory = Path("dayData")  # adjust path as needed
output_path = Path("timeseries_updated.json")

json_files = sorted(sensor_data_directory.glob("*.json")) 

if not json_files:
    raise FileNotFoundError(f"No JSON files found in {sensor_data_directory}")

js_all = {}

for num, file_path in enumerate(tqdm(json_files, desc="Combining JSON files", unit="file")):
    with open(file_path, "r", encoding="utf-8") as f:
        js = json.load(f)

    if num == 0:
        js_all['STATION'] = js.get('STATION', [])
    else:
        current_stats = {v['STID']: k for k, v in enumerate(js_all['STATION'])}

        for stat in js.get('STATION', []):
            if stat['STID'] in current_stats:
                stid = current_stats[stat['STID']]
                for field, data in js_all['STATION'][stid].items():
                    if isinstance(data, dict) and field in stat:
                        if field == 'OBSERVATIONS':
                            for k in js_all['STATION'][stid][field]:
                                if k in stat[field]:
                                    js_all['STATION'][stid][field][k] += stat[field][k]
                                else:
                                    js_all['STATION'][stid][field][k] += [None for _ in range(len(stat[field]['date_time']))]
                        else:
                            js_all['STATION'][stid][field].update(stat[field])
            else:
                js_all['STATION'].append(stat)

# Save the combined JSON file
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(js_all, file)

print(f"\nCombined {len(json_files)} files into {output_path}")
