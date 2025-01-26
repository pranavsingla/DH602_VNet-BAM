import os

def rename_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate over each file
    for file_name in files:
        # Check if the file matches the desired format
        if file_name.startswith("spleen_") and file_name.endswith(".npz"):
            # Extract the slice number from the file name
            slice_number = file_name.split("_")[2].split(".")[0]
            # Construct the new file name
            new_file_name = f"spleen_slice{slice_number}.npz"
            # Rename the file
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
            print(f"Renamed {file_name} to {new_file_name}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing the files: ")
    rename_files(folder_path)
