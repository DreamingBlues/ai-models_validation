# Downloads station data and compiles all days into one file, sectioning by network
# Abtin Olaee 2025

import os
import json
import shutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    "token": "18c00d44ddf644809b61d733618d578f",
    "start_time": "2025-01-01 00:00:00",
    "end_time": "2025-01-11 00:00:00",
    "networks": [1, 2, 229, 231],
    "bbox": "-124.50, 32.30, -114.00, 42.00",
    "units": "metric",
    "temp_dir": "./sensorData/temp/",
    "output_file": "./sensorData/situ_data.json",
    
    "cleanup": True,  # delete 'temp_dir' after completion
}

# Helpsers
def synoptic_api_request(config):
    "Downloads synoptic station files based on CONFIG"
    # Parse in config parameters
    token = config["token"]
    start_time = datetime.strptime(config["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(config["end_time"], "%Y-%m-%d %H:%M:%S")
    networks = config["networks"]
    
    # create temporary folder to download each time step for each network
    download_dir = Path(config["temp_dir"])
    download_dir.mkdir(parents=True, exist_ok=True) 

    # time formatting function for download parameters
    def fmt(dt):
        return dt.strftime("%Y%m%d%H%M") 

    current = start_time
    print(f"Downloading to {config['temp_dir']}...")

    while current < end_time:
        day_begin = current
        day_end = current.replace(hour=23, minute=59)

        for net in networks:
            params = {
                "token":token,
                "start": fmt(day_begin),
                "end": fmt(day_end),
                "obtimezone": "utc",
                "units": config["units"],
                "bbox": config["bbox"],
                "networks": str(net),
                "sensorvars": 1,
                "complete": 1,
                "sitinghistory":1,
                "timeformat": "%Y%m%d%H%M%S",
            }

            url =  "https://api.synopticdata.com/v2/stations/timeseries"
            out_file = download_dir / f"{net}-{day_begin.strftime('%Y%m%d')}.json"

            # send out download requests
            try:
                r=requests.get(url, params=params, timeout=120)
                r.raise_for_status()
                data = r.json()

                if data.get("SUMMARY", {}).get("RESPONSE_CODE", 1) != 1:
                    raise RuntimeError(data.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "Unknown API error"))
                
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                
                print(f"Saved {out_file.name}")

            except Exception as e:
                print(f"ERROR!!! Network {net} on {day_begin:%Y-%m-%d}: {e}")
                continue
            
        # move counter to next day 
        current += timedelta(days=1)



def compile_json_data(config):
    "Compiles all of the temporary JSON files into one big JSON file"
    print(f"\nCompiling into {config['output_file']}")
    input_dir = Path(config['temp_dir'])
    output_path = Path(config['output_file'])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print("No files found!")
        return False

    js_all = {}

    for num, file_path in enumerate(tqdm(json_files, desc="Compiling Files", unit="file")):
        with open(file_path, "r", encoding = 'utf-8') as f:
            js = json.load(f)

        if num == 0:
            js_all["STATION"] = js.get("STATION", [])
        else:
            current_stats = {v["STID"]: k for k, v in enumerate(js_all["STATION"])}

            for stat in js.get("STATION", []):
                stid = stat["STID"]

                if stid not in current_stats: # if new station
                    js_all["STATION"].append(stat)
                
                else: #  station alr present
                    i = current_stats[stid]

                    # merge observations
                    for field, data in stat.get("OBSERVATIONS", {}).items():
                        if field in js_all["STATION"][i]["OBSERVATIONS"]:
                            js_all["STATION"][i]["OBSERVATIONS"][field] += data
                        else:
                            js_all["STATION"][i]["OBSERVATIONS"][field] = data

                    # add metadata
                    if "SENSOR_VARIABELS" in stat:
                        js_all["STATION"][i]["SENSOR_VARIABLES"] = stat["SENSOR_VARIABLES"]

                    for meta_key in ["UNITS", "ELEVATION", "ELEV_DEM", "PERIOD_OF_RECORD"]:
                        if meta_key in stat:
                            js_all["STATION"][i][meta_key] = stat[meta_key]


    with open(output_path,"w", encoding="utf-8") as file:
        json.dump(js_all, file)            

    print(f"Final file saved to {output_path}")
    return True




def cleanup(config):
    """Remove the temporary file used for individual downloads"""

    temp_path =Path(config["temp_dir"])
    if temp_path.exists() and config["cleanup"]:
        shutil.rmtree(temp_path)
        print("Cleanup complete")






def main():
    synoptic_api_request(CONFIG)
    success = compile_json_data(CONFIG)
    if success:
        cleanup(CONFIG)

if __name__ == "__main__":
    main()
