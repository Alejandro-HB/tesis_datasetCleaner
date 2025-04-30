import os
import pandas as pd

def merge_csv_files(input_folder, output_file):
    # List to store dataframes
    dataframes = []

    # Iterate over all CSV files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            print(f'Reading {filename}')
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f'Merged CSV saved to {output_file}')

# Example usage
input_folder = './tripAdvisor/tlaxcala'  # Folder containing the CSVs
output_file = './tripAdvisor/tlaxcala/tlaxcala_all.csv'
merge_csv_files(input_folder, output_file)