# AI Models Validation

## Overview

This repository contains a Python-based workflow for validating AI and numerical weather prediction (NWP) model forecasts against station-based surface observations. The project focuses on evaluating 10-meter wind speed forecasts during the January 7–10, 2025 Santa Ana wind event over California and the greater Los Angeles region.

The workflow compares AI-based weather models, including Aurora, FourCastNet v2, FourCastNet v3, and GraphCast, against traditional or operational forecast products such as ECMWF IFS and the National Blend of Models (NBM). Forecasts are evaluated against processed Synoptic station observations using temporal, spatial, and station-network-based validation methods.

## Project Goals

* Compare AI and NWP wind-speed forecasts against station-derived observations.
* Evaluate model performance across multiple forecast lead times.
* Analyze temporal error, spatial error, signed bias, and station-network differences.
* Generate figures and tables for model validation during a high-impact Santa Ana wind event.

## Models Included

| Model Key   | Model Name               |
| ----------- | ------------------------ |
| `ifs`       | ECMWF IFS                |
| `nbm`       | National Blend of Models |
| `fcn2`      | FourCastNet v2           |
| `fcn3`      | FourCastNet v3           |
| `aurora`    | Aurora                   |
| `graphcast` | GraphCast                |

## Main Data Inputs

The main station observation file used by the analysis scripts is:

```text
./sensorData/station_data.nc
```

The main station-observation variable is:

```text
ws_10m_corr
```

The main model forecast variable is:

```text
wind_speed
```

Processed model forecast files are expected to follow this structure:

```text
./{model}/processed_data/{model}_processed_CA_Day{day}.nc
```

Example:

```text
./ifs/processed_data/ifs_processed_CA_Day05.nc
```

## Main Scripts

| Script                       | Purpose                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| `temporal_leadtimes.py`      | Plots regional mean wind speed across forecast lead times         |
| `temporal_bias.py`           | Plots regional mean bias across forecast lead times               |
| `temporal_error_table.py`    | Creates CA/LA temporal metric tables                              |
| `spatial_error_dashboard.py` | Computes and plots spatial RMSE, MAE, MAPE, and Pearson r         |
| `spatial_map_error.py`       | Maps station-derived wind speed and model-minus-observation error |
| `error_vs_station_and_z0.py` | Compares grid-cell error against station count and `z0_era5`      |
| `network_station_map.py`     | Maps station locations by network                                 |
| `network_observations.py`    | Plots station-observed wind speed by network                      |


## Quick Start

Run the main temporal comparison:

```bash
python temporal_leadtimes.py --models ifs nbm graphcast aurora fcn2 fcn3
```

Generate the temporal bias figure:

```bash
python temporal_bias.py --models ifs nbm graphcast aurora fcn2 fcn3
```

Generate the temporal error metric table:

```bash
python temporal_error_table.py --models ifs nbm graphcast aurora fcn2 fcn3
```

Generate spatial error dashboards:

```bash
python spatial_error_dashboard.py --region CA --models ifs nbm graphcast aurora fcn2 fcn3
python spatial_error_dashboard.py --region LA --models ifs nbm graphcast aurora fcn2 fcn3
```

Generate the 72-hour spatial wind-speed error map:

```bash
python spatial_map_error.py --region LA --day 05
```

## Documentation

More detailed documentation is split into separate files:

* [Data description](docs/data.md)
* [Script reference](docs/scripts.md)
* [Full analysis workflow](docs/workflow.md)

## Notes

* All times are handled in UTC.
* The main verification period is January 7–10, 2025.
* The project currently focuses on deterministic wind-speed forecast validation.
* Empty model grid cells without station observations are excluded from spatial comparison.
* Figures and tables are saved under the `figs/` directory.

