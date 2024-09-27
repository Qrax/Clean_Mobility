# Functions

## `plot_data`
This repository includes a `plot_data` function that allows flexible plotting with 2D/3D and trendline options.

### Usage

```python
plot_data(data, x_col, y_col, z_col=None, plot_type='scatter', trendline=None, degree=1, plot_z_as='heatmap')
```

### Parameters

| **Parameter**   | **Description**                                      | **Possible Values**                                  | **Default**     | **Example**                            |
|-----------------|------------------------------------------------------|------------------------------------------------------|-----------------|----------------------------------------|
| `data`          | DataFrame containing the data                        | Any pandas DataFrame                                 | N/A             | `data_file_filtered`                   |
| `x_col`         | Column name for X-axis                               | String (column name)                                 | N/A             | `'Latitude'`                           |
| `y_col`         | Column name for Y-axis                               | String (column name)                                 | N/A             | `'Longitude'`                          |
| `z_col`         | Column name for Z-axis (optional)                    | String (column name) or `None`                       | `None`          | `'Speed over ground in km/h'`          |
| `plot_type`     | Type of plot                                         | `'scatter'`, `'line'`                                | `'scatter'`     | `'line'`                               |
| `trendline`     | Adds a trendline to the 2D plot (optional)           | `'linear'`, `'polynomial'`, `None`                   | `None`          | `'linear'`, `'polynomial'`             |
| `degree`        | Degree of the polynomial trendline (if `polynomial`) | Integer (e.g., `1`, `2`, `3`)                        | `1`             | `2`                                    |
| `plot_z_as`     | How to plot Z-axis data if provided                  | `'3d'` (3D plot), `'heatmap'` (2D plot with heatmap) | `'heatmap'`     | `'3d'`                                 |

### Example

```python
plot_data(data_file_filtered, 'Latitude', 'Longitude', 'Speed over ground in km/h', plot_type='scatter', trendline='polynomial', degree=2, plot_z_as='3d')
```

---

## `ImporteerKolomNamenDataBase`
This function imports the column names from a reference CSV file.

### Usage

```python
ImporteerKolomNamenDataBase(bestandsnaam='column_namen_referentie_bestand.csv')
```

### Parameters

| **Parameter**   | **Description**                                | **Possible Values**                           | **Default**                             | **Example**                       |
|-----------------|------------------------------------------------|-----------------------------------------------|-----------------------------------------|-----------------------------------|
| `bestandsnaam`  | Name of the reference CSV file to import        | String (file path)                            | `'column_namen_referentie_bestand.csv'` | `'custom_column_file.csv'`        |

### Example

```python
ImporteerKolomNamenDataBase(bestandsnaam='my_columns.csv')
```

---

## `KolomnamenDataBase`
This function fetches the column names for a specific format header from a database.

### Usage

```python
KolomnamenDataBase(format_header, bestandsnaam='column_namen_referentie_bestand')
```

### Parameters

| **Parameter**   | **Description**                                | **Possible Values**                          | **Default**                             | **Example**                       |
|-----------------|------------------------------------------------|----------------------------------------------|-----------------------------------------|-----------------------------------|
| `format_header` | The format header for which to fetch column names | String                                      | N/A                                     | `'my_format_header'`              |
| `bestandsnaam`  | Name of the reference CSV file to search         | String (file path)                           | `'column_namen_referentie_bestand'`     | `'custom_column_file.csv'`        |

### Example

```python
KolomnamenDataBase(format_header='my_format_header', bestandsnaam='my_columns.csv')
```

---

## `KolomNamenJuistZetten`
This function adjusts column names in a DataFrame based on a format header from the database.

### Usage

```python
KolomNamenJuistZetten(dataframe, debug=False)
```

### Parameters

| **Parameter**   | **Description**                                | **Possible Values**                          | **Default** | **Example**                       |
|-----------------|------------------------------------------------|----------------------------------------------|-------------|-----------------------------------|
| `dataframe`     | The DataFrame to adjust the column names for    | A pandas DataFrame                          | N/A         | `my_dataframe`                   |
| `debug`         | Print debug information                        | Boolean                                     | `False`     | `True`                            |

### Example

```python
KolomNamenJuistZetten(my_dataframe, debug=True)
```

---

## `DataInladen`
This function loads data from a file into a pandas DataFrame and adjusts its column names based on the format header.

### Usage

```python
DataInladen(directory_data, debug=False)
```

### Parameters

| **Parameter**   | **Description**                                | **Possible Values**                          | **Default** | **Example**                       |
|-----------------|------------------------------------------------|----------------------------------------------|-------------|-----------------------------------|
| `directory_data`| The file path to load the data from             | String (file path)                           | N/A         | `'path/to/data.csv'`             |
| `debug`         | Print debug information                        | Boolean                                     | `False`     | `True`                            |

### Example

```python
DataInladen('data/my_data.csv', debug=True)
```


---

## `resample_and_merge`
This function resamples two dataframes based on a specified time frequency and merges them on their time columns.

### Usage

```python
resample_and_merge(df1_n, df2_n, freq='1S', time_column_df1='Dataloggertijd, in s', time_column_df2='Dataloggertijd, in s')
```

### Parameters

| **Parameter**       | **Description**                                              | **Possible Values**                           | **Default**               | **Example**                       |
|---------------------|--------------------------------------------------------------|-----------------------------------------------|---------------------------|-----------------------------------|
| `df1_n`             | The first DataFrame to resample and merge                    | A pandas DataFrame                            | N/A                       | `df1`                            |
| `df2_n`             | The second DataFrame to resample and merge                   | A pandas DataFrame                            | N/A                       | `df2`                            |
| `freq`              | Frequency for resampling the data                            | String (e.g., `'1S'` for 1 second)            | `'1S'`                    | `'5T'`                           |
| `time_column_df1`   | Name of the time column in the first DataFrame               | String (column name)                          | `'Dataloggertijd, in s'`   | `'Time'`                         |
| `time_column_df2`   | Name of the time column in the second DataFrame              | String (column name)                          | `'Dataloggertijd, in s'`   | `'Time'`                         |

### Example

```python
resample_and_merge(df1, df2, freq='1T', time_column_df1='Time', time_column_df2='Time')
```
