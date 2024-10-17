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
    low_memory = True
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

import tkinter as tk
from tkinter import ttk

import tkinter as tk
from tkinter import ttk

def plot_window_gui(df):
    # Create the pop-up window
    window = tk.Tk()
    window.title("Plot Data Configuration")
    window.geometry("1000x460")  # Make the window bigger for better visibility

    # Labels and dropdowns for x, y, and z axes
    tk.Label(window, text="Select X-axis", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    x_axis = ttk.Combobox(window, values=df.columns.to_list(), width=50, font=("Arial", 14))  # Larger width and font
    x_axis.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(window, text="Select Y-axis", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10, sticky="w")
    y_axis = ttk.Combobox(window, values=df.columns.to_list(), width=50, font=("Arial", 14))  # Larger width and font
    y_axis.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(window, text="Select Z-axis (optional)", font=("Arial", 14)).grid(row=2, column=0, padx=10, pady=10, sticky="w")
    z_axis = ttk.Combobox(window, values=[None] + df.columns.to_list(), width=50, font=("Arial", 14))  # Larger width and font
    z_axis.grid(row=2, column=1, padx=10, pady=10)

    # Plot type dropdown
    tk.Label(window, text="Select Plot Type", font=("Arial", 14)).grid(row=3, column=0, padx=10, pady=10, sticky="w")
    plot_type = ttk.Combobox(window, values=['scatter', 'line'], width=50, font=("Arial", 14))  # Larger width and font
    plot_type.set('scatter')  # Default value
    plot_type.grid(row=3, column=1, padx=10, pady=10)

    # Trendline type dropdown
    tk.Label(window, text="Select Trendline", font=("Arial", 14)).grid(row=4, column=0, padx=10, pady=10, sticky="w")
    trendline = ttk.Combobox(window, values=[None, 'linear', 'polynomial'], width=50, font=("Arial", 14))  # Larger width and font
    trendline.grid(row=4, column=1, padx=10, pady=10)

    # Polynomial degree slider (only for polynomial trendlines)
    tk.Label(window, text="Polynomial Degree (for polynomial trendline)", font=("Arial", 14)).grid(row=5, column=0, padx=10, pady=10, sticky="w")
    degree = tk.Scale(window, from_=1, to=5, orient=tk.HORIZONTAL, font=("Arial", 14))  # Larger font
    degree.grid(row=5, column=1, padx=10, pady=10)

    # Z-axis plot type
    tk.Label(window, text="Plot Z axis as", font=("Arial", 14)).grid(row=6, column=0, padx=10, pady=10, sticky="w")
    plot_z_as = ttk.Combobox(window, values=['heatmap', '3d'], width=50, font=("Arial", 14))  # Larger width and font
    plot_z_as.grid(row=6, column=1, padx=10, pady=10)

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

    # Button to plot the data with significantly increased width and height
    plot_button = tk.Button(window, text="Plot", command=plot_button_action, width=20, height=2, font=("Arial", 14))  # Big button with larger font
    plot_button.grid(row=7, column=0, columnspan=2, padx=10, pady=20)

    window.mainloop()

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

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def variable_selector(df):
    """
    Create a pop-up window to select columns for x_as, y_as, and z_as,
    and add a button to copy the selection code for reuse.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    tuple: The selected columns for x_as, y_as, and z_as.
    """
    # Create a pop-up window
    window = tk.Tk()
    window.title("Select Variables")
    window.geometry("900x250")  # Make the window bigger for better visibility

    # Variables to store the selections
    x_as = tk.StringVar()
    y_as = tk.StringVar()
    z_as = tk.StringVar()

    # Dropdown for x_as
    tk.Label(window, text="Select X-axis:", font=("Arial", 14)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    x_dropdown = ttk.Combobox(window, textvariable=x_as, width=50, font=("Arial", 14))
    x_dropdown['values'] = list(df.columns)
    x_dropdown.grid(row=0, column=1, padx=10, pady=10)

    # Dropdown for y_as
    tk.Label(window, text="Select Y-axis:", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=10, sticky="w")
    y_dropdown = ttk.Combobox(window, textvariable=y_as, width=50, font=("Arial", 14))
    y_dropdown['values'] = list(df.columns)
    y_dropdown.grid(row=1, column=1, padx=10, pady=10)

    # Dropdown for z_as (optional)
    tk.Label(window, text="Select Z-axis (optional):", font=("Arial", 14)).grid(row=2, column=0, padx=10, pady=10,
                                                                                sticky="w")
    z_dropdown = ttk.Combobox(window, textvariable=z_as, width=50, font=("Arial", 14))
    z_dropdown['values'] = [None] + list(df.columns)
    z_dropdown.grid(row=2, column=1, padx=10, pady=10)

    # To store selected values
    selected_values = [None, None, None]

    # Function to capture the selections and close the window
    def set_values():
        selected_values[0] = x_as.get()
        selected_values[1] = y_as.get()
        selected_values[2] = z_as.get()
        window.destroy()  # Close the window once selections are made

    # Function to display the code to get selected values
    def copy_code():
        selected_x = x_as.get() or 'None'
        selected_y = y_as.get() or 'None'
        selected_z = z_as.get() or 'None'

        # Python code for selecting the variables
        code = f"x_as = '{selected_x}'\n"
        code += f"y_as = '{selected_y}'\n"
        code += f"z_as = '{selected_z}'\n"

        # Display the code in a message box for copying
        code_window = tk.Toplevel(window)
        code_window.title("Generated Code")
        code_window.geometry("500x200")

        code_label = tk.Label(code_window, text="Copy the code below:", font=("Arial", 12))
        code_label.pack(padx=10, pady=10)

        code_text = tk.Text(code_window, height=6, width=60, font=("Arial", 12))
        code_text.insert(tk.END, code)
        code_text.pack(padx=10, pady=10)

        # Automatically select the code in the text box
        code_text.focus()
        code_text.tag_add("selectall", "1.0", tk.END)
        code_text.tag_config("selectall", background="lightyellow")

    # "Set" button to confirm selections, make it much larger
    set_button = tk.Button(window, text="Set", command=set_values, width=20, height=2, font=("Arial", 14))
    set_button.grid(row=3, column=0, padx=10, pady=20)

    # "Copy Code" button to generate the code, much larger
    copy_code_button = tk.Button(window, text="Copy Code", command=copy_code, width=20, height=2, font=("Arial", 14))
    copy_code_button.grid(row=3, column=1, padx=10, pady=20)

    # Run the window loop
    window.mainloop()

    # Return the selected values for x_as, y_as, and z_as
    return selected_values[0], selected_values[1], selected_values[2]

def Get_directory(base_directory):
    # Dropdown widgets voor elke stap
    voertuig_dropdown = widgets.Dropdown(description='Voertuig:')
    jaartal_dropdown = widgets.Dropdown(description='Jaartal:')
    situatie_dropdown = widgets.Dropdown(description='Situatie:')
    dataset_dropdown = widgets.Dropdown(description='Dataset:')

    # Confirm-knop
    confirm_button = widgets.Button(description="Map Laden", button_style='success')

    # Output widget voor berichten of fouten
    output = widgets.Output()

    # Variabele voor het opslaan van de geselecteerde directory
    directory = None


    def update_jaartal_options(change):
        """Update de jaartal dropdown op basis van het gekozen voertuig."""
        selected_voertuig = voertuig_dropdown.value
        voertuig_path = os.path.join(base_directory, selected_voertuig)

        if os.path.exists(voertuig_path):
            # Lijst van beschikbare jaartallen (mappen) in de voertuigmap
            jaartal_options = [name for name in os.listdir(voertuig_path) if
                               os.path.isdir(os.path.join(voertuig_path, name))]
            jaartal_dropdown.options = jaartal_options
        else:
            jaartal_dropdown.options = []
            situatie_dropdown.options = []
            dataset_dropdown.options = []


    def update_situatie_options(change):
        """Update de situatie dropdown op basis van het gekozen jaartal."""
        selected_voertuig = voertuig_dropdown.value
        selected_jaartal = jaartal_dropdown.value
        jaartal_path = os.path.join(base_directory, selected_voertuig, selected_jaartal)

        if os.path.exists(jaartal_path):
            # Lijst van beschikbare situaties (mappen) in de jaartalmap
            situatie_options = [name for name in os.listdir(jaartal_path) if
                                os.path.isdir(os.path.join(jaartal_path, name))]
            situatie_dropdown.options = situatie_options
        else:
            situatie_dropdown.options = []
            dataset_dropdown.options = []


    def update_dataset_options(change):
        """Update de dataset dropdown op basis van de gekozen situatie."""
        selected_voertuig = voertuig_dropdown.value
        selected_jaartal = jaartal_dropdown.value
        selected_situatie = situatie_dropdown.value
        situatie_path = os.path.join(base_directory, selected_voertuig, selected_jaartal, selected_situatie)

        if os.path.exists(situatie_path):
            # Lijst van beschikbare datasets (mappen) in de situatiemap
            dataset_options = [name for name in os.listdir(situatie_path) if
                               os.path.isdir(os.path.join(situatie_path, name))]
            dataset_dropdown.options = dataset_options
        else:
            dataset_dropdown.options = []


    def read_directory(b):
        """Geeft het pad naar de geselecteerde datasetmap en slaat het op."""
        global directory  # Gebruik de globale variabele om het pad op te slaan
        selected_voertuig = voertuig_dropdown.value
        selected_jaartal = jaartal_dropdown.value
        selected_situatie = situatie_dropdown.value
        selected_dataset = dataset_dropdown.value

        if selected_voertuig and selected_jaartal and selected_situatie and selected_dataset:
            # Maak het volledige pad naar de geselecteerde dataset
            directory = os.path.join(base_directory, selected_voertuig, selected_jaartal, selected_situatie,
                                     selected_dataset)
            with output:
                output.clear_output()
                print(f"Geselecteerde map opgeslagen als 'directory': {directory}")
        else:
            with output:
                output.clear_output()
                print("Selecteer alle opties (voertuig, jaartal, situatie, dataset) om verder te gaan.")


    # Verbind de veranderingen in de dropdowns aan hun respectievelijke update functies
    voertuig_dropdown.observe(update_jaartal_options, names='value')
    jaartal_dropdown.observe(update_situatie_options, names='value')
    situatie_dropdown.observe(update_dataset_options, names='value')

    # Koppel de 'Confirm'-knop aan de read_directory functie
    confirm_button.on_click(read_directory)

    # Begin met het ophalen van de voertuigen (eerste dropdown)
    voertuig_dropdown.options = [name for name in os.listdir(base_directory) if
                                 os.path.isdir(os.path.join(base_directory, name))]

    # Toon de widgets
    display(voertuig_dropdown, jaartal_dropdown, situatie_dropdown, dataset_dropdown, confirm_button, output)









