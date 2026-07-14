# README: station_data.nc
# Abtin Olaee 2026

## Overview

This file (`station_data.nc`) contains point-based surface meteorological observations derived from the Synoptic API. The raw JSON station observations have been processed into a `station x time` NetCDF structure for analysis, model validation, and forecast comparison.

The dataset is temporally aligned, meaning all station observations have been snapped to a common 6-hour UTC time grid.

This file also includes wind-speed correction fields used to standardize observed wind speed to a common 10 m reference height.

---

## Source Data

The input data are downloaded from the Synoptic API for the California study region.

Included station networks:

| Network ID | Network |
|---:|---|
| `1` | ASOS/AWOS |
| `2` | RAWS |
| `229` | PG&E |
| `231` | SCE |

---

## Dataset Structure

| Coordinate | Dimension | Description |
|---|---|---|
| `station` | `(station)` | Station ID |
| `time` | `(time)` | 6-hour UTC timestamp |
| `latitude` | `(station)` | Station latitude |
| `longitude` | `(station)` | Station longitude |
| `elevation` | `(station)` | Station elevation |
| `network` | `(station)` | Synoptic network ID |

---

## Data Variables

Most meteorological variables are stored as 2D arrays with shape:

```text
(station, time)
```

Missing values are represented as `NaN`.

Primary time-varying variables may include:

| Variable | Dimension | Description |
|---|---|---|
| `air_temp` | `(station, time)` | Air temperature |
| `wind_speed` | `(station, time)` | Original observed wind speed |
| `wind_direction` | `(station, time)` | Wind direction |
| `relative_humidity` | `(station, time)` | Relative humidity |
| `pressure` | `(station, time)` | Pressure |
| `dew_point_temperature` | `(station, time)` | Dew point temperature |
| `ws_10m_corr` | `(station, time)` | Wind speed corrected to 10 m |

Station-level correction variables include:

| Variable | Dimension | Description |
|---|---|---|
| `height_agl` | `(station)` | Wind sensor height above ground level, in meters |
| `height_agl_source` | `(station)` | Source used for sensor height |
| `z0_era5` | `(station)` | ERA5 surface roughness sampled at the nearest grid cell, in meters |

Possible `height_agl_source` values:

| Value | Meaning |
|---|---|
| `json` | Sensor height was read from Synoptic metadata |
| `pge_csv` | Sensor height was read from `pge_agl.csv` |
| `network_fallback` | Sensor height was assigned using network default |
| `missing` | No valid sensor height was available |

---

## Wind-Speed Height Correction

Observed wind speeds are corrected to a common 10 m reference height using a logarithmic wind profile:

```text
ws_10m_corr = wind_speed * ln(10 / z0) / ln(z / z0)
```

where:

| Symbol | Meaning |
|---|---|
| `wind_speed` | Original observed wind speed |
| `z` | Sensor height above ground level |
| `z0` | ERA5 surface roughness |
| `ws_10m_corr` | Corrected 10 m wind speed |

ERA5 surface roughness is read from:

```text
./sensorData/era5_z0.nc
```

The ERA5 `z0` field is sampled at the nearest grid cell to each station location.

---

## Sensor Height Assumptions

Sensor height is assigned in the following order:

1. Use Synoptic JSON sensor-position metadata if available.
2. For PG&E stations, use `pge_agl.csv` if available.
3. If no metadata are available, use network-level fallback assumptions.

Fallback values:

| Network ID | Network | Assumed Height AGL |
|---:|---|---:|
| `1` | ASOS/AWOS | `10.0 m` |
| `2` | RAWS | `6.1 m` |
| `229` | PG&E | `6.1 m` |
| `231` | SCE | `7.62 m` |

---

## Data Processing

### 1. Time Alignment

Observations are aligned to a 6-hour time grid:

```text
00:00, 06:00, 12:00, 18:00 UTC
```

Each observation is matched to the nearest grid time using a tolerance of:

```text
+/- 30 minutes
```

If no observation is available within that window, the value is saved as `NaN`.

---

### 2. Variable Cleaning

Synoptic variable suffixes are removed for consistency.

Examples:

```text
air_temp_set_1      -> air_temp
wind_speed_set_1    -> wind_speed
```

Non-numeric or dictionary-style quality-control fields are excluded.

Examples of excluded fields:

```text
QC_SUMMARY
qc_summary
weather_cond_code
cloud_layer_1_code
cloud_layer_2_code
cloud_layer_3_code
```

---

### 3. AGL and Roughness Correction

After the station time series is created, the script adds:

```text
height_agl
height_agl_source
z0_era5
ws_10m_corr
```

These fields allow wind-speed observations from different station networks to be compared on a common 10 m basis.

---

## Example Dataset Structure

Example xarray structure:

```text
<xarray.Dataset>
Dimensions:           (station: N, time: 41)

Coordinates:
  * station           (station) object ...
  * time              (time) datetime64[ns] ...
    latitude          (station) float64 ...
    longitude         (station) float64 ...
    elevation         (station) float64 ...
    network           (station) int/float ...

Data variables:
    air_temp          (station, time) float32 ...
    wind_speed        (station, time) float32 ...
    wind_direction    (station, time) float32 ...
    pressure          (station, time) float32 ...
    height_agl        (station) float32 ...
    height_agl_source (station) object/string ...
    z0_era5           (station) float32 ...
    ws_10m_corr       (station, time) float32 ...
```

---

## Python Usage Examples

### 1. Load the Dataset

```python
import xarray as xr

ds = xr.open_dataset("./sensorData/station_data.nc", engine="h5netcdf")
print(ds)
```

---

### 2. Select One Station

```python
station = ds.sel(station="KSFO")

station["wind_speed"].plot()
station["ws_10m_corr"].plot()
```

---

### 3. Filter Stations by Region

```python
min_lat, max_lat = 32.5, 42.0
min_lon, max_lon = -124.5, -114.0

mask = (
    (ds.latitude >= min_lat) & (ds.latitude <= max_lat) &
    (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
)

ca_stations = ds.isel(station=mask)

print(ca_stations)
```

---

### 4. Filter by Network

```python
# PG&E stations
pge = ds.where(ds.network == 229, drop=True)

print(pge)
```

---

### 5. Compare Original and Corrected Wind Speed

```python
station = ds.sel(station="KSFO")

station["wind_speed"].plot(label="Original")
station["ws_10m_corr"].plot(label="Corrected to 10 m")
```

---

## Notes

* All timestamps are treated as UTC.
* Units follow the Synoptic API metric output where available.
* Variable-specific units should be checked before analysis.
* `wind_speed` is the original station-reported wind speed.
* `ws_10m_corr` is the wind speed adjusted to 10 m using station AGL and ERA5 surface roughness.
* The dataset is intended for station-based model validation and gridded forecast comparison.