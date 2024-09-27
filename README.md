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

---

## `launch_plot_window`
This repository includes a `launch_plot_window` function that creates an interactive GUI for configuring and plotting data with 2D/3D options and trendline selection.

### Usage

```python
plot_window_gui(df)
```

### Parameters

| **Parameter**   | **Description**                                      | **Possible Values**                                  | **Default**     | **Example**                            |
|-----------------|------------------------------------------------------|------------------------------------------------------|-----------------|----------------------------------------|
| `df`            | DataFrame containing the data                        | Any pandas DataFrame                                 | N/A             | `data_file_filtered`                   |

### GUI Elements

| **Element**      | **Description**                                      | **Possible Values**                                  | **Default**     | **Example**                            |
|------------------|------------------------------------------------------|------------------------------------------------------|-----------------|----------------------------------------|
| `x_axis`         | Dropdown to select the X-axis column                 | String (column name)                                 | N/A             | `'Latitude'`                           |
| `y_axis`         | Dropdown to select the Y-axis column                 | String (column name)                                 | N/A             | `'Longitude'`                          |
| `z_axis`         | Dropdown to select the Z-axis column (optional)      | String (column name) or `None`                       | `None`          | `'Speed over ground in km/h'`          |
| `plot_type`      | Dropdown to select the plot type                     | `'scatter'`, `'line'`                                | `'scatter'`     | `'line'`                               |
| `trendline`      | Dropdown to add a trendline to the plot (optional)   | `'linear'`, `'polynomial'`, `None`                   | `None`          | `'linear'`, `'polynomial'`             |
| `degree`         | Slider to select the degree of the polynomial trendline | Integer (e.g., `1`, `2`, `3`)                        | `1`             | `2`                                    |
| `plot_z_as`      | Dropdown to select how to handle the Z-axis          | `'3d'` (3D plot), `'heatmap'` (2D plot with heatmap) | `'heatmap'`     | `'3d'`                                 |
| `plot_button`    | Button to trigger the plot                           | N/A                                                  | N/A             | Press to generate the plot             |

### Example

```python
plot_window_gui(data_file_filtered)
```

Once the GUI is launched, users can select columns, plot types, trendlines, and other options through the dropdowns and sliders, then press the "Plot" button to visualize the data.

---

## `variable_selector`
This repository includes a `variable_selector` function that opens a pop-up window, allowing users to interactively select columns for the X-axis, Y-axis, and optionally Z-axis from a DataFrame.

### Usage

```python
x_as, y_as, z_as = variable_selector(df)
```

### Parameters

| **Parameter**   | **Description**                                      | **Possible Values**                                  | **Default**     | **Example**                            |
|-----------------|------------------------------------------------------|------------------------------------------------------|-----------------|----------------------------------------|
| `df`            | DataFrame containing the data                        | Any pandas DataFrame                                 | N/A             | `merged_df`                            |

### Returns

| **Return Value** | **Description**                                      | **Type**                                              | **Example**                            |
|------------------|------------------------------------------------------|-------------------------------------------------------|----------------------------------------|
| `x_as`           | Selected X-axis column                               | String (column name)                                  | `'Latitude'`                           |
| `y_as`           | Selected Y-axis column                               | String (column name)                                  | `'Longitude'`                          |
| `z_as`           | Selected Z-axis column (optional)                    | String (column name) or `None`                        | `'Speed over ground in km/h'`          |

### GUI Elements

| **Element**      | **Description**                                      | **Possible Values**                                  | **Default**     | **Example**                            |
|------------------|------------------------------------------------------|------------------------------------------------------|-----------------|----------------------------------------|
| `x_as`           | Dropdown to select the X-axis column                 | String (column name)                                 | None            | `'Latitude'`                           |
| `y_as`           | Dropdown to select the Y-axis column                 | String (column name)                                 | None            | `'Longitude'`                          |
| `z_as`           | Dropdown to select the Z-axis column (optional)      | String (column name) or `None`                       | None            | `'Speed over ground in km/h'`          |
| `Set` Button     | Button to confirm and store the selected columns     | N/A                                                  | N/A             | Press to confirm selections            |

### Example

```python
# Assuming df is your DataFrame
x_as, y_as, z_as = variable_selector(df)

# Now use the selected columns for plotting or other operations
print(f"Selected X: {x_as}, Y: {y_as}, Z: {z_as}")

# For example, you can call your plot function:
plot_data(filtered_df, x_as, y_as, z_as)
```

### Example Output:

After selecting the columns and pressing "Set" in the pop-up window, the selected columns will be stored in the variables `x_as`, `y_as`, and `z_as`:

```python
Selected X: 'Latitude'
Selected Y: 'Longitude'
Selected Z: 'Speed over ground in km/h'
```

These variables can then be used directly in your functions for plotting or other analysis.

---

---

## `resample_and_merge_multiple`

This function resamples and merges multiple dataframes based on a specified time frequency. It processes each dataframe by converting a time column to a datetime index, resampling numeric and non-numeric columns appropriately, and then merges all resampled dataframes on their indices.

### Usage

```python
merged_df = resample_and_merge_multiple(dfs, freq='1S', time_column='Dataloggertijd, in s')
```

### Parameters

| **Parameter**   | **Description**                                           | **Possible Values**              | **Default**                | **Example**                       |
|-----------------|-----------------------------------------------------------|----------------------------------|----------------------------|-----------------------------------|
| `dfs`           | List of DataFrames to resample and merge                  | List of pandas DataFrames        | N/A                        | `[df1, df2, df3]`                 |
| `freq`          | Frequency for resampling the data                         | String (e.g., `'1S'`, `'5T'`)    | `'1S'`                     | `'5T'`                            |
| `time_column`   | Name of the time column in each DataFrame                 | String (column name)             | `'Dataloggertijd, in s'`   | `'Time'`                          |

### Returns

| **Return Value** | **Description**                                           | **Type**              |
|------------------|-----------------------------------------------------------|-----------------------|
| `merged_df`      | Merged DataFrame after resampling                         | pandas DataFrame      |

### Example

```python
# Assuming dfs is a list of DataFrames you want to resample and merge
merged_df = resample_and_merge_multiple(dfs, freq='1T', time_column='Time')
```

---

## `DataUitzoekenGui`

This function creates an interactive GUI for selecting and loading multiple CSV files from a specified directory. It allows users to select multiple files, load them into DataFrames, resample and merge them using `resample_and_merge_multiple`, and provides feedback through a status label.

### Usage

```python
result = DataUitzoekenGui(directory, freq='1S')
```

### Parameters

| **Parameter**   | **Description**                                    | **Possible Values**              | **Default**                | **Example**                       |
|-----------------|----------------------------------------------------|----------------------------------|----------------------------|-----------------------------------|
| `directory`     | The directory path containing the CSV files        | String (directory path)          | N/A                        | `'data/csv_files'`                |
| `freq`          | Frequency for resampling the data                  | String (e.g., `'1S'`, `'5T'`)    | `'1S'`                     | `'5T'`                            |

### Returns

| **Return Value** | **Description**                                      | **Type**              |
|------------------|------------------------------------------------------|-----------------------|
| `result`         | Dictionary containing the merged DataFrame           | `{'merged_df': DataFrame}` |

### GUI Elements

- **File Selector**: A multi-select widget listing all `.csv` files in the specified directory.
- **Load Button**: A button to load and merge the selected files.
- **Status Label**: A label to display status messages.

### Example

```python
# Launch the GUI for selecting and merging CSV files
result = DataUitzoekenGui('data/csv_files', freq='1T')

# After loading, access the merged DataFrame
merged_df = result.get('merged_df')

# Use the merged DataFrame as needed
print(merged_df.head())
```

---
