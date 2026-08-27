import os
import json
import csv

file_path = 'last_frame_params.csv'
with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
    # Step 2: Create a CSV reader object
    csv_reader = csv.reader(csvfile)

    params_dict = {}

    # Step 3: Loop through each row in the CSV file
    for row in csv_reader:
        # Skip the header row
        if row[0] == 'W_idx':
            continue
        
        widx = int(row[0])
        cl_tup = (int(row[2]), int(row[3]))
        if widx not in params_dict.keys():
            params_dict[widx] = [cl_tup]
        else:
            params_dict[widx].append(cl_tup)
            
    print(params_dict)
