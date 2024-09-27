import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import copy
from math import sqrt,ceil
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from numpy.polynomial import Polynomial
import tkinter as tk
from tkinter import ttk
import ipywidgets as widgets

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
    on_bad_lines='skip',
    low_memory = False
    )
    df = KolomNamenJuistZetten(df,debug)
    return df


# Update the function to use the correct time columns based on the observed data
def resample_and_merge(df1_n, df2_n, freq='1S', time_column_df1='Dataloggertijd, in s',
                       time_column_df2='Dataloggertijd, in s'):
    df1 = df1_n.copy()
    df2 = df2_n.copy()

    # Convert the time columns (which are in seconds) to a numeric format, forcing errors to NaN
    df1['Indextijd'] = pd.to_numeric(df1[time_column_df1], errors='coerce')
    df2['Indextijd'] = pd.to_numeric(df2[time_column_df2], errors='coerce')

    # Drop rows with NaN values in the time column
    df1.dropna(subset=['Indextijd'], inplace=True)
    df2.dropna(subset=['Indextijd'], inplace=True)

    # Convert the time columns (which are now numeric) to a timedelta format
    df1['Indextijd'] = pd.to_timedelta(df1['Indextijd'], unit='s')
    df2['Indextijd'] = pd.to_timedelta(df2['Indextijd'], unit='s')

    # Round the time columns to the nearest second (or desired frequency)
    df1['Indextijd'] = df1['Indextijd'].dt.round(freq)
    df2['Indextijd'] = df2['Indextijd'].dt.round(freq)

    # Set the time columns as the index
    df1.set_index('Indextijd', inplace=True)
    df2.set_index('Indextijd', inplace=True)

    # Separate numeric and non-numeric columns
    df1_numeric = df1.select_dtypes(include=np.number)
    df1_non_numeric = df1.select_dtypes(exclude=np.number)

    df2_numeric = df2.select_dtypes(include=np.number)
    df2_non_numeric = df2.select_dtypes(exclude=np.number)

    # Resample numeric columns (mean)
    df1_resampled_numeric = df1_numeric.resample(freq).mean()
    df2_resampled_numeric = df2_numeric.resample(freq).mean()

    # Resample non-numeric columns (using 'first' or 'last' or another aggregation method)
    df1_resampled_non_numeric = df1_non_numeric.resample(freq).first()  # Or use 'last' or another method
    df2_resampled_non_numeric = df2_non_numeric.resample(freq).first()

    # Combine resampled numeric and non-numeric dataframes
    df1_resampled = pd.concat([df1_resampled_numeric, df1_resampled_non_numeric], axis=1)
    df2_resampled = pd.concat([df2_resampled_numeric, df2_resampled_non_numeric], axis=1)

    # Merge the two dataframes based on the time index
    merged_df = pd.merge(df1_resampled, df2_resampled, left_index=True, right_index=True, how='outer', suffixes=('_df1', '_df2'))

    return merged_df



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
    x = pd.to_numeric(data[x_col], errors='coerce').values
    y = pd.to_numeric(data[y_col], errors='coerce').values
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
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, color='red', label='Linear Trendline')

            # Add the equation and R-squared value to the plot
            equation_text = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}'
            ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        elif trendline == 'polynomial':
            try:
                # Perform polynomial regression
                p = Polynomial.fit(x, y, degree)
                y_fit = p(x)
                ax.plot(x, y_fit, color='red', label=f'Polynomial Trendline (degree {degree})')

                # Calculate R-squared value for polynomial regression
                residuals = y - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # Add the polynomial equation and R-squared value to the plot
                coef = p.convert().coef  # Get the polynomial coefficients in standard form
                poly_eq = " + ".join([f"{coef[i]:.3f}x^{i}" if i > 0 else f"{coef[i]:.3f}" for i in range(len(coef))])
                equation_text = f'y = {poly_eq}\nR² = {r_squared:.3f}'
                ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

            except np.linalg.LinAlgError as e:
                print(f"Error fitting polynomial trendline: {e}")

        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    # Check if a label exists for the legend, and add it
    if ax.get_legend_handles_labels()[1]:  # Check if there are labels
        ax.legend()

    plt.show()

def plot_window(data, x_col, y_col, z_col=None, plot_type='scatter', trendline=None, degree=1, plot_z_as='heatmap'):
    """
    Plots the data based on the provided x, y, and optional z axis columns. Supports 2D and 3D plotting.
    """
    # Extract data and remove NaN/Inf values
    x = pd.to_numeric(data[x_col], errors='coerce').values
    y = pd.to_numeric(data[y_col], errors='coerce').values
    mask = np.isfinite(x) & np.isfinite(y)

    if z_col is not None and z_col in data.columns:
        z = pd.to_numeric(data[z_col], errors='coerce').values
        mask &= np.isfinite(z)
        z = z[mask]

    x = x[mask]
    y = y[mask]

    fig = plt.figure(figsize=(8, 8))

    if z_col is not None and z_col in data.columns and plot_z_as == '3d':
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        if plot_type == 'scatter':
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
            plt.colorbar(scatter, ax=ax, pad=0.1)
        elif plot_type == 'line':
            ax.plot(x, y, z)

        ax.set_title(f'{x_col} vs {y_col} vs {z_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)

    elif z_col is not None and z_col in data.columns and plot_z_as == 'heatmap':
        # 2D plot with heatmap (color for z-axis)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=10)
        plt.colorbar(scatter, ax=ax, pad=0.1)

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
            equation_text = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}'
            ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        elif trendline == 'polynomial':
            try:
                p = Polynomial.fit(x, y, degree)
                y_fit = p(x)
                ax.plot(x, y_fit, color='red', label=f'Polynomial Trendline (degree {degree})')
                residuals = y - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
                coef = p.convert().coef
                poly_eq = " + ".join([f"{coef[i]:.3f}x^{i}" if i > 0 else f"{coef[i]:.3f}" for i in range(len(coef))])
                equation_text = f'y = {poly_eq}\nR² = {r_squared:.3f}'
                ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

            except np.linalg.LinAlgError as e:
                print(f"Error fitting polynomial trendline: {e}")

        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    ax.legend()
    plt.show()

from ipywidgets import widgets
from IPython.display import display

# Aangepaste resample_and_merge functie om meerdere dataframes te kunnen verwerken
def resample_and_merge_multiple(dfs, freq='1S', time_column='Dataloggertijd, in s'):
    resampled_dfs = []

    for df in dfs:
        df = df.copy()
        # Convert the time column to numeric format
        df['Indextijd'] = pd.to_numeric(df[time_column], errors='coerce')
        # Drop rows with NaN in the time column
        df.dropna(subset=['Indextijd'], inplace=True)
        # Convert to timedelta
        df['Indextijd'] = pd.to_timedelta(df['Indextijd'], unit='s')
        # Round the time column to the nearest frequency
        df['Indextijd'] = df['Indextijd'].dt.round(freq)
        # Set as index
        df.set_index('Indextijd', inplace=True)
        # Separate numeric and non-numeric columns
        df_numeric = df.select_dtypes(include=np.number)
        df_non_numeric = df.select_dtypes(exclude=np.number)
        # Resample
        df_resampled_numeric = df_numeric.resample(freq).mean()
        df_resampled_non_numeric = df_non_numeric.resample(freq).first()
        # Combine
        df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1)
        resampled_dfs.append(df_resampled)

    # Merge all dataframes on the index
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), resampled_dfs)
    return merged_df

def DataUitzoekenGui(directory,freq='1S'):
    # Haal de lijst van bestanden op in de directory (optioneel filteren op .csv bestanden)
    files_in_directory = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Maak een selectiemenu voor de bestanden
    file_selector = widgets.SelectMultiple(
        options=files_in_directory,
        description='Bestanden:',
        disabled=False
    )

    # Label om statusberichten te tonen
    status_label = widgets.Label(value='')

    # Hier definiëren we een mutable object om de returnwaarde op te slaan
    result = {}

    # Functie om de geselecteerde bestanden in te laden en te mergen
    def load_files(b):
        print('data aant laden')
        selected_files = file_selector.value
        if not selected_files:
            status_label.value = 'Geen bestanden geselecteerd.'
            return

        dataframes = []
        for file_name in selected_files:
            file_path = os.path.join(directory, file_name)
            data = DataInladen(file_path, debug=False)
            dataframes.append(data)
            print(f"{file_name} is geladen.")

        # Merge de dataframes
        merged_df = resample_and_merge_multiple(dataframes,freq=freq)
        print("Alle dataframes zijn samengevoegd.")

        # Sla het samengevoegde dataframe op in het result dict
        result['merged_df'] = merged_df

        status_label.value = 'Bestanden geladen en samengevoegd.'

    # Knop om de bestanden te laden
    load_button = widgets.Button(
        description='Bestanden Laden',
        disabled=False,
        button_style='success',
        tooltip='Klik om de geselecteerde bestanden te laden en samen te voegen',
        icon='check'
    )

    # Koppel de functie aan de knop
    load_button.on_click(load_files)

    # Toon de widgets
    display(file_selector)
    display(load_button)
    display(status_label)

    # Return het result dict zodat je er buiten de functie bij kunt
    return result


def launch_plot_window(df):
    # Create the pop-up window
    window = tk.Tk()
    window.title("Plot Data Configuration")

    # Labels and dropdowns for x, y, and z axes
    tk.Label(window, text="Select X-axis").grid(row=0, column=0)
    x_axis = ttk.Combobox(window, values=df.columns.to_list())
    x_axis.grid(row=0, column=1)

    tk.Label(window, text="Select Y-axis").grid(row=1, column=0)
    y_axis = ttk.Combobox(window, values=df.columns.to_list())
    y_axis.grid(row=1, column=1)

    tk.Label(window, text="Select Z-axis (optional)").grid(row=2, column=0)
    z_axis = ttk.Combobox(window, values=[None] + df.columns.to_list())
    z_axis.grid(row=2, column=1)

    # Plot type dropdown
    tk.Label(window, text="Select Plot Type").grid(row=3, column=0)
    plot_type = ttk.Combobox(window, values=['scatter', 'line'])
    plot_type.set('scatter')  # Default value
    plot_type.grid(row=3, column=1)

    # Trendline type dropdown
    tk.Label(window, text="Select Trendline").grid(row=4, column=0)
    trendline = ttk.Combobox(window, values=[None, 'linear', 'polynomial'])
    trendline.grid(row=4, column=1)

    # Polynomial degree slider (only for polynomial trendlines)
    tk.Label(window, text="Polynomial Degree (for polynomial trendline)").grid(row=5, column=0)
    degree = tk.Scale(window, from_=1, to=5, orient=tk.HORIZONTAL)
    degree.grid(row=5, column=1)

    # Z-axis plot type
    tk.Label(window, text="Plot Z axis as").grid(row=6, column=0)
    plot_z_as = ttk.Combobox(window, values=['heatmap', '3d'])
    plot_z_as.grid(row=6, column=1)

    def plot_button_action():
        x_col = x_axis.get()
        y_col = y_axis.get()
        z_col = z_axis.get() if z_axis.get() != '' and z_axis.get() != 'None' else None
        plot_type_val = plot_type.get()
        trendline_val = trendline.get()
        degree_val = degree.get()
        plot_z_as_val = plot_z_as.get()

        if x_col and y_col:
            # Call the plot_data function if both x and y columns are selected
            plot_data(df, x_col, y_col, z_col, plot_type_val, trendline_val, degree_val, plot_z_as_val)
        else:
            print("Please select both X and Y columns.")

    # Button to plot the data
    plot_button = tk.Button(window, text="Plot", command=plot_button_action)
    plot_button.grid(row=7, column=0, columnspan=2)

    window.mainloop()


import ipywidgets as widgets
from IPython.display import display


import ipywidgets as widgets
from IPython.display import display

import ipywidgets as widgets
from IPython.display import display

import ipywidgets as widgets
from IPython.display import display

# Global variables to store selected values
x_as = None
y_as = None
z_as = None

import ipywidgets as widgets
from IPython.display import display

# Declare the global variables initially
x_as = None
y_as = None
z_as = None

import ipywidgets as widgets
from IPython.display import display


def variable_selector(df):
    """
    Create a pop-up window to select columns for x_as, y_as, and z_as.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    tuple: The selected columns for x_as, y_as, and z_as.
    """
    # Create a pop-up window
    window = tk.Tk()
    window.title("Select Variables")

    # Variables to store the selections
    x_as = tk.StringVar()
    y_as = tk.StringVar()
    z_as = tk.StringVar()

    # Dropdown for x_as
    tk.Label(window, text="Select X-axis:").grid(row=0, column=0)
    x_dropdown = ttk.Combobox(window, textvariable=x_as)
    x_dropdown['values'] = list(df.columns)
    x_dropdown.grid(row=0, column=1)

    # Dropdown for y_as
    tk.Label(window, text="Select Y-axis:").grid(row=1, column=0)
    y_dropdown = ttk.Combobox(window, textvariable=y_as)
    y_dropdown['values'] = list(df.columns)
    y_dropdown.grid(row=1, column=1)

    # Dropdown for z_as (optional)
    tk.Label(window, text="Select Z-axis (optional):").grid(row=2, column=0)
    z_dropdown = ttk.Combobox(window, textvariable=z_as)
    z_dropdown['values'] = [None] + list(df.columns)
    z_dropdown.grid(row=2, column=1)

    # To store selected values
    selected_values = [None, None, None]

    # Function to capture the selections and close the window
    def set_values():
        selected_values[0] = x_as.get()
        selected_values[1] = y_as.get()
        selected_values[2] = z_as.get()
        window.destroy()  # Close the window once selections are made

    # "Set" button to confirm selections
    set_button = tk.Button(window, text="Set", command=set_values)
    set_button.grid(row=3, column=0, columnspan=2)

    # Run the window loop
    window.mainloop()

    # Return the selected values for x_as, y_as, and z_as
    return selected_values[0], selected_values[1], selected_values[2]







