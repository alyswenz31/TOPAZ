#!/bin/bash

# Define the target directory where you want to copy the folders
TARGET_DIR="CrockerToDownload"

# List of source directories
SOURCE_DIRS=("ChangeW_00" "ChangeW_05" "NoChangeW_00" "NoChangeW_05")

# List of parameter combinations (Cidx, Lidx, Widx)
pars_idc=("17 3 0" "6 24 0" "19 0 0" "8 5 0" "4 4 0" "1 14 0" "14 6 0" "24 24 0" "19 14 0" "11 6 0" "17 3 5" "6 24 5" "19 0 5" "8 5 5" "4 4 5" "1 14 5" "14 6 5" "24 24 5" "19 14 5" "11 6 5")

# Check if the target directory exists, create it if not
if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory does not exist. Creating it now..."
    mkdir -p "$TARGET_DIR"
fi

# Loop through each combination of parameters
for pars_idx in "${pars_idc[@]}"; do
    # Unpack the parameters into Cidx, Lidx, Widx
    Cidx=$(echo $pars_idx | cut -d ' ' -f 1)
    Lidx=$(echo $pars_idx | cut -d ' ' -f 2)
    Widx=$(echo $pars_idx | cut -d ' ' -f 3)

    # Create the folder for the specific pars_idc
    pars_folder="$TARGET_DIR/CLW_${Cidx}_${Lidx}_${Widx}"
    mkdir -p "$pars_folder"
    
    # Check the Widx to determine which group of source directories to use
    if [[ $Widx -eq 0 ]]; then
        # For Widx ending in 0, group ChangeW_00 and NoChangeW_00
        GROUP=("ChangeW_00" "NoChangeW_00")
    elif [[ $Widx -eq 5 ]]; then
        # For Widx ending in 5, group ChangeW_05 and NoChangeW_05
        GROUP=("ChangeW_05" "NoChangeW_05")
    else
        echo "Widx is neither 0 nor 5 for pars_idc: $pars_idx. Skipping."
        continue
    fi

    # Loop through each relevant source directory in the group
    for SOURCE in "${GROUP[@]}"; do
        echo "Processing source directory: $SOURCE for pars_idc: $pars_idx"

        # Construct the folder path dynamically
        FOLDER="Simulated_Grid/ODE_Align/Cidx_$(printf "%02d" $Cidx)_Lidx_$(printf "%02d" $Lidx)_Widx_$(printf "%02d" $Widx)/"

        # Check if the folder exists in the source directory
        if [ -d "$SOURCE/$FOLDER" ]; then
            echo "Copying files from $SOURCE/$FOLDER to $pars_folder"

            # Only copy the specified files: ABC_crocker and true_crocker
            for file in ABC_crocker* true_crocker*; do
                if [ -f "$SOURCE/$FOLDER/$file" ]; then
                    cp "$SOURCE/$FOLDER/$file" "$pars_folder/"
                    echo "Copied $file from $SOURCE/$FOLDER to $pars_folder"
                else
                    echo "$file not found in $SOURCE/$FOLDER"
                fi
            done
        else
            echo "Folder $FOLDER not found in $SOURCE"
        fi
    done
done

echo "Folder copy operation complete."
