#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:55:26 2023

@author: linzilin
"""

import os
import pandas as pd

def remove_repetitive_rows(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Drop duplicate rows from the DataFrame
    df.drop_duplicates(inplace=True)

    # Save the cleaned data back to the file
    df.to_csv(file_path, index=False)

def clean_csv_files_in_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            remove_repetitive_rows(file_path)

# Replace 'folder_path' with the path to your folder containing the CSV files
folder_path = 'CHOSEN_faces-clustering-output-eps=0.4/facepaths copy'
clean_csv_files_in_folder(folder_path)

def count_rows(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Count the number of rows in the DataFrame
    num_rows = len(df)

    return num_rows

def rank_csv_files_by_rows(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Create a dictionary to store the file names and their corresponding row counts
    file_row_counts = {}

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            num_rows = count_rows(file_path)
            file_row_counts[file] = num_rows

    # Sort the dictionary by row counts in descending order
    sorted_file_row_counts = {k: v for k, v in sorted(file_row_counts.items(), key=lambda item: item[1], reverse=True)}

    return sorted_file_row_counts

def save_ranking_to_csv(folder_path, ranking_file):
    ranked_files = rank_csv_files_by_rows(folder_path)

    # Create a DataFrame from the ranked_files dictionary
    df = pd.DataFrame(list(ranked_files.items()), columns=['File', 'Number of Rows'])

    # Save the DataFrame to a new CSV file
    df.to_csv(ranking_file, index=False)

# Replace 'folder_path' with the path to your folder containing the CSV files
folder_path = 'CHOSEN_faces-clustering-output-eps=0.4/facepaths unique'
ranking_file = 'ranking.csv'
save_ranking_to_csv(folder_path, ranking_file)

# Print the ranked files
#for i, (file, num_rows) in enumerate(ranked_files.items(), 1):
#    print(f"Rank {i}: {file} - Number of rows: {num_rows}")

