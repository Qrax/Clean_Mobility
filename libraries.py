import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import copy
from math import sqrt,ceil
from scipy.stats import linregress

def ImporteerKolomNamenDataBase(bestandsnaam='column_namen_referentie_bestand.csv'):
    try:
        df = pd.read_csv(bestandsnaam, delimiter=',', comment='#')
        return df
    except FileNotFoundError:
        print(f"Bestand {bestandsnaam} niet gevonden.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Fout bij het parseren van het CSV-bestand {bestandsnaam}.")
        return pd.DataFrame()


def KolomnamenDataBase(format_header, bestandsnaam='column_namen_referentie_bestand'):
    df = ImporteerKolomNamenDataBase(bestandsnaam)
    if df.empty:
        raise ValueError("De kolomnaamendatabase is leeg of kon niet geladen worden.")

    row = df[df['format_header'] == format_header]
    if row.empty:
        raise ValueError(f"Format header '{format_header}' niet gevonden in de database.")

    kolomnamen_str = row.iloc[0]['kolomnamen']
    kolomnamen = [kolom.strip() for kolom in kolomnamen_str.split(';')]
    return kolomnamen

def KolomNamenJuistZetten(dataframe,debug=False):
    if debug:
        print(dataframe.iloc[0,2])
        print(dataframe.head())
    kolomnamen = KolomnamenDataBase(dataframe.iloc[0,2])
    if debug:
        print(kolomnamen)
        print(len(kolomnamen))
        for i in range(len(kolomnamen)):
            print(i+1,kolomnamen[i])
    dataframe.columns = kolomnamen[0:-1]
    return dataframe

def DataInladen(directory_data,debug=False):
    # Data inladen direct als DataFrame en kolomnamen aanpassen
    df = pd.read_csv(
    directory_data,
    delimiter=',',
    encoding='latin1',
    comment="#",
    on_bad_lines='warn'
    )
    df = KolomNamenJuistZetten(df,debug)
    return df

def flat_plot(data_file, height = 'Snelheid over de grond in km/h'):
    plt.figure(figsize=(6,6))
    plt.scatter(data_file['Latitude'], data_file['Longitude'], c=data_file[height], cmap='viridis')
    plt.title('GPS data vs speed')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')

    # Adding a colorbar with a label for the speed in km/h
    cbar = plt.colorbar()
    cbar.set_label('Speed (km/h)')
    plt.grid()
    plt.show()
    return None


# Update the function to use the correct time columns based on the observed data
def resample_and_merge(df1_n, df2_n, freq='1S', time_column_df1='Dataloggertijd, in s',
                       time_column_df2='Dataloggertijd, in s'):
    df1 = df1_n.copy()
    df2 = df2_n.copy()

    # Convert the time columns (which are in seconds) to a numeric format, forcing errors to NaN
    df1[time_column_df1] = pd.to_numeric(df1[time_column_df1], errors='coerce')
    df2[time_column_df2] = pd.to_numeric(df2[time_column_df2], errors='coerce')

    # Drop rows with NaN values in the time column
    df1.dropna(subset=[time_column_df1], inplace=True)
    df2.dropna(subset=[time_column_df2], inplace=True)

    # Convert the time columns (which are now numeric) to a timedelta format
    df1[time_column_df1] = pd.to_timedelta(df1[time_column_df1], unit='s')
    df2[time_column_df2] = pd.to_timedelta(df2[time_column_df2], unit='s')

    # Round the time columns to the nearest second (or desired frequency)
    df1[time_column_df1] = df1[time_column_df1].dt.round(freq)
    df2[time_column_df2] = df2[time_column_df2].dt.round(freq)

    # Set the time columns as the index
    df1.set_index(time_column_df1, inplace=True)
    df2.set_index(time_column_df2, inplace=True)

    # Resample both dataframes to the desired frequency (1 second by default), ensuring only numeric columns are aggregated
    df1_resampled = df1.resample(freq).mean(numeric_only=True)
    df2_resampled = df2.resample(freq).mean(numeric_only=True)

    # Merge the two dataframes based on the time index
    merged_df = pd.merge(df1_resampled, df2_resampled, left_index=True, right_index=True, how='outer')

    return merged_df


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
from numpy.polynomial import Polynomial


def plot_data(data, x_col, y_col, z_col=None, plot_type='scatter', trendline=None, degree=1, plot_z_as='heatmap'):
    """
    Plots the data based on the provided x, y, and optional z axis columns. Supports 2D and 3D plotting.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x_col: Name of the column for the X-axis.
    - y_col: Name of the column for the Y-axis.
    - z_col: Name of the column for the Z-axis (optional, if not provided, a 2D plot is generated).
    - plot_type: Type of plot ('scatter', 'line'). Default is 'scatter'.
    - trendline: Type of trendline ('linear', 'polynomial'). Default is None.
    - degree: Degree of the polynomial trendline (if applicable). Default is 1.
    - plot_z_as: How to handle the Z-axis if it's provided ('3d' for a 3D plot, 'heatmap' for a 2D scatter with heatmap). Default is 'heatmap'.
    """
    # Extract data and remove NaN/Inf values
    x = data[x_col].values
    y = data[y_col].values
    mask = np.isfinite(x) & np.isfinite(y)

    if z_col is not None:
        z = data[z_col].values
        mask &= np.isfinite(z)
        z = z[mask]

    x = x[mask]
    y = y[mask]

    fig = plt.figure(figsize=(8, 8))

    if z_col is not None and plot_z_as == '3d':
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')

        if plot_type == 'scatter':
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(f'{z_col}')
        elif plot_type == 'line':
            ax.plot(x, y, z)

        ax.set_title(f'{x_col} vs {y_col} vs {z_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)

    elif z_col is not None and plot_z_as == 'heatmap':
        # 2D plot with heatmap (color for z-axis)
        ax = fig.add_subplot(111)

        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=10)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(f'{z_col}')

        ax.set_title(f'{x_col} vs {y_col} (Color: {z_col})')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    else:
        # Regular 2D plot
        ax = fig.add_subplot(111)

        if plot_type == 'scatter':
            ax.scatter(x, y, c='b', s=10)
        elif plot_type == 'line':
            ax.plot(x, y)

        # Add trendline if requested
        if trendline == 'linear':
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, color='red', label='Linear Trendline')
        elif trendline == 'polynomial':
            try:
                p = Polynomial.fit(x, y, degree)
                ax.plot(x, p(x), color='red', label=f'Polynomial Trendline (degree {degree})')
            except np.linalg.LinAlgError as e:
                print(f"Error fitting polynomial trendline: {e}")

        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    # Check if a label exists for the legend, and add it
    if ax.get_legend_handles_labels()[1]:  # Check if there are labels
        ax.legend()

    plt.show()




