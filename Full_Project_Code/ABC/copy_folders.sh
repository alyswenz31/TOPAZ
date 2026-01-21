#!/bin/bash

# Define the target directory where you want to copy the folders
TARGET_DIR="FilesToDownload"

# List of source directories
SOURCE_DIRS=("ChangeW_00" "ChangeW_05" "NoChangeW_00" "NoChangeW_05")

# List of parameter combinations (Cidx, Lidx, Widx)
# pars_idc=((17 3 0) (6 24 0) (19 0 0) (8 5 0) (4 4 0) (1 14 0) (14 6 0) (24 24 0) (19 14 0) (17 3 5) (6 24 5) (19 0 5) (8 5 5) (4 4 5) (1 14 5) (14 6 5) (24 24 5) (19 14 5))
pars_idc=("17 3 0" "6 24 0" "19 0 0" "8 5 0" "4 4 0" "1 14 0" "14 6 0" "24 24 0" "19 14 0" "17 3 5" "6 24 5" "19 0 5" "8 5 5" "4 4 5" "1 14 5" "14 6 5" "24 24 5" "19 14 5")

# Check if the target directory exists, create it if not
if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory does not exist. Creating it now..."
    mkdir -p "$TARGET_DIR"
fi

# Loop through each source directory
for SOURCE in "${SOURCE_DIRS[@]}"; do
    echo "Processing source directory: $SOURCE"

    # Loop through each combination of parameters
    for pars_idx in "${pars_idc[@]}"; do
        # Unpack the parameters into Cidx, Lidx, Widx
        Cidx=$(echo $pars_idx | cut -d ' ' -f 1)
        Lidx=$(echo $pars_idx | cut -d ' ' -f 2)
        Widx=$(echo $pars_idx | cut -d ' ' -f 3)

        # Construct the folder path dynamically
        FOLDER="Simulated_Grid/ODE_Align/Cidx_$(printf "%02d" $Cidx)_Lidx_$(printf "%02d" $Lidx)_Widx_$(printf "%02d" $Widx)/"

        # Check if the folder exists in the source directory
        if [ -d "$SOURCE/$FOLDER" ]; then
            echo "Copying folder $FOLDER from $SOURCE to $TARGET_DIR"
            cp -r "$SOURCE/$FOLDER" "$TARGET_DIR/$SOURCE"
        else
            echo "Folder $FOLDER not found in $SOURCE"
        fi
    done
done

echo "Folder copy operation complete."
