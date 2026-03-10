# Downloads station data and compiles all days into one file, sectioning by network
# Abtin Olaee 2025

import os
import json
import shutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "token": " *INSERT API TOKEN HERE* ",
    "start_time": "2025-01-01 00:00:00",
    "end_time": "2025-01-11 00:00:00",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "networks": [1, 2, 229, 231],
    "bbox": "-124.409591,32.534156,-114.131211,42.009518",
    "units": "metric",
    "temp_dir": "./sensorData/temp/",
    "output_file": "./sensorData/situ_data.json",
    "cleanup_temp": True,  # delete temp_dir after completion
    "timeout": 120
}



# =============================================================================
# HELPERS
# =============================================================================

def synoptic_api_request(config):
    """Downloads daily JSON files based on CONFIG."""
    token = config["token"]
    start_dt = datetime.strptime(config["start_time"], config["date_format"])
    # adding one day to end_time to ensure the final day is included in the loop
    end_dt_exclusive = datetime.strptime(config["end_time"], config["date_format"]) + timedelta(days=1)
    networks = config["networks"]
    
    download_dir = Path(config["temp_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)

    def fmt(dt):
        return dt.strftime("%Y%m%d%H%M")

    curr = start_dt
    print(f"Downloading to {config['temp_dir']}...")
    
    while curr < end_dt_exclusive:
        day_begin = curr
        day_end = curr.replace(hour=23, minute=59)

        for net in networks:
            params = {
                "token": token,
                "start": fmt(day_begin),
                "end": fmt(day_end),
                "obtimezone": "utc",
                "units": config["units"],
                "bbox": config["bbox"],
                "networks": str(net),
            }
            url = "https://api.synopticdata.com/v2/stations/timeseries"
            out_file = download_dir / f"{net}-{day_begin.strftime('%Y%m%d')}.json"

            try:
                r = requests.get(url, params=params, timeout=config["timeout"])
                r.raise_for_status()
                data = r.json()
                
                if data.get("SUMMARY", {}).get("RESPONSE_CODE", 1) != 1:
                    raise RuntimeError(data.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "Unknown API error"))
                
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                print(f"Saved {out_file.name}")

            except Exception as e:
                print(f"[ERROR] Network {net} on {day_begin:%Y-%m-%d}: {e}")
                continue

        curr += timedelta(days=1)

def compile_json_data(config):
    """Combines temp JSON files into the final output_file."""
    print(f"\nCompiling to {config['output_file']}")
    input_dir = Path(config["temp_dir"])
    output_path = Path(config["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print("No files found to compile.")
        return False

    js_all = {}

    for num, file_path in enumerate(tqdm(json_files, desc="Merging Files", unit="file")):
        with open(file_path, "r", encoding="utf-8") as f:
            js = json.load(f)

        if num == 0:
            js_all['STATION'] = js.get('STATION', [])
        else:
            current_stats = {v['STID']: k for k, v in enumerate(js_all['STATION'])}

            for stat in js.get('STATION', []):
                stid = stat['STID']
                if stid in current_stats:
                    idx = current_stats[stid]
                    for field, data in stat.get('OBSERVATIONS', {}).items():
                        if field in js_all['STATION'][idx]['OBSERVATIONS']:
                            js_all['STATION'][idx]['OBSERVATIONS'][field] += data
                        else:
                            js_all['STATION'][idx]['OBSERVATIONS'][field] = data
                else:
                    js_all['STATION'].append(stat)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(js_all, file)

    print(f"Success! Final file saved to: {output_path}")
    return True

def cleanup(config):
    """Deletes the temporary directory."""
    temp_path = Path(config["temp_dir"])
    if temp_path.exists() and config["cleanup_temp"]:
        print(f"\nCleaning up temporary files in {config['temp_dir']}...")
        shutil.rmtree(temp_path)
        print("\nCleanup complete.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    synoptic_api_request(CONFIG)
    success = compile_json_data(CONFIG)
    if success:
        cleanup(CONFIG)

if __name__ == "__main__":
    main()