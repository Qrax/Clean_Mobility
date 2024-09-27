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

def KolomNamenJuistZetten(dataframe):
    #print(df.iloc[0,2])
    #print(df.head())
    kolomnamen = KolomnamenDataBase(dataframe.iloc[0,2])
    #print(kolomnamen)
    #print(len(kolomnamen))
    #for i in range(len(kolomnamen)):
    #    print(i+1,kolomnamen[i])
    dataframe.columns = kolomnamen[0:-1]
    return dataframe

def DataInladen(directory_data):
    # Data inladen direct als DataFrame en kolomnamen aanpassen
    df = pd.read_csv(
    directory_data,
    delimiter=',',
    encoding='latin1',
    comment="#",
    on_bad_lines='skip'
    )
    df = KolomNamenJuistZetten(df)
    return df


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

