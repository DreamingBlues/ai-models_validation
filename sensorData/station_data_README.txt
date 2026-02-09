# README: station_data.nc
# Abtin Olaee 2025

## Overview
This file (`station_data.nc`) contains point-based surface meteorological observations derived from the Synoptic API. It has been processed into a **2D Orthogonal Array (Station × Time)** format for efficient analysis, machine learning, and model validation.

Unlike the raw JSON source, this dataset is **temporally aligned**, meaning all stations have been snapped to a unified 6-hourly UTC time grid.

## File Structure
The file follows the **CF-Conventions for Discrete Sampling Geometries** (Point Data).
It contains meteorological station networks IDs, 1, 2, 229, 231
ID 1: ASOS/AWOS (Automated Surface/Weather Observing Systems)
ID 2: RAWS (Remote Automated Weather Stations)
ID 229: PGE (Pacific Gas and Electric) 
ID 231: SCE (Southern California Edison)

* **Format:** NetCDF-4 (HDF5)
* **Dimensions:**
    * `station`: ~4024 (Number of unique stations)
    * `time`: 41 (Time steps from Jan 1 – Jan 11, 2025)

## Coordinates (Metadata)
These variables define the "rows" and "columns" of the data matrix.

| Coordinate Name | Dimension   | Type       | Description                         |
| :---            | :---        | :---       | :---                                |
| **station**     | `(station)` | `string`   | Unique Station ID (`'KAAT'`, ...)   |
| **time**        | `(time)`    | `datetime` | UTC Timestamps (00, 06, 12, 18 UTC) |
| **latitude**    | `(station)` | `float32`  | Station Latitude (WGS84)            |
| **longitude**   | `(station)` | `float32`  | Station Longitude (WGS84)           |
| **elevation**   | `(station)` | `float32`  | Station Elevation (meters)          |

## Data Variables
Each variable is a 2D matrix of shape `(station, time)`.
* **Missing Data:** Represented as `NaN` (floating-point null).
* **Units:** Metric (Standard Synoptic output: °C, m/s, %, Pascal).

**Primary Variables:**
* `air_temp`: Air Temperature (°C)
* `wind_speed`: Wind Speed (m/s)
* `wind_direction`: Wind Direction (Degrees 0-360)
* `relative_humidity`: Relative Humidity (%)
* `pressure`: Station Pressure (Pa)
* `dew_point_temperature`: Dew Point (°C)

*(Note: The file contains all numeric variables found in the raw source, typically ~40-50 variables total.)*

## Dataset Structure
<xarray.Dataset>
Dimensions:      (station: 4024, time: 41)
Coordinates:
  * station      (station) object 'KAAT' 'KSFO' 'KLAX' ...  <-- The Keys (Row Labels)
  * time         (time) datetime64[ns] 2025-01-01 ...       <-- The Columns
    latitude     (station) float64 41.48 37.62 33.94 ...    <-- Metadata (1D)
    longitude    (station) float64 -120.5 -122.4 -118.4 ... <-- Metadata (1D)
    elevation    (station) float64 1200.5 5.2 34.0 ...      <-- Metadata (1D)
Data variables:
    air_temp     (station, time) float32 15.2 15.5 ...      <-- The "Tabs" (2D)
    wind_speed   (station, time) float32 3.5 4.1 ...        <-- The "Tabs" (2D)
    pressure     (station, time) float32 1013 1012 ...      <-- The "Tabs" (2D)


## Data Processing & QA
1.  **Time Alignment:**
    * **Resolution:** 6 Hours (00:00, 06:00, 12:00, 18:00 UTC).
    * **Method:** Observations were snapped to the nearest master grid time using a tolerance of **±30 minutes**.
    * **Data Loss:** Timesteps with no observations within 30 minutes of the target hour are marked as `NaN`.
2.  **Cleaning:**
    * Suffixes like `_set_1` or `_set_2d` were stripped to standardize variable names (e.g., `air_temp_set_1` → `air_temp`).
    * Dictionary-based or non-numeric QC columns (e.g., `QC_SUMMARY`) were excluded to maintain a strictly numeric dataset.

## Python Usage Examples

### 1. Loading the Data
```python
import xarray as xr

# Open dataset
ds = xr.open_dataset("station_data_2d.nc", engine="h5netcdf")
print(ds)
```

### 2. Selecting a Specific Station (by ID)
```python 
# Get time series for San Francisco Int'l (KSFO)
ksfo = ds.sel(station="KSFO")

# Plot Temperature
ksfo["air_temp"].plot()
```

### 3. Filtering by Region (e.g., California Box)
Since stations are not on a grid, you filter using boolean masks on the coordinate arrays.
```python
# Define Box
min_lat, max_lat = 32.5, 42.0
min_lon, max_lon = -124.5, -114.0

# Create Mask
mask = (
    (ds.latitude >= min_lat) & (ds.latitude <= max_lat) &
    (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
)

# Apply Mask
ca_stations = ds.isel(station=mask)
print(f"Found {ca_stations.sizes['station']} stations in CA.")
```

