import os
import csv
import argparse
import numpy as np

def create_dataset_csv(txt_dir, img_dir, output_csv, idx_offset=-15, frame_range=4):
    # List to store rows for csv
    dataset = []
    
    # 1. Iterate over the directory containing the .mp4.txt files.
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.mp4.txt'):
            file_base_name = txt_file.replace('.mp4.txt', '')
            
            # Check for "Aim-Assist ON" in the filename
            aim_assist = 1 if "Aim-Assist ON" in txt_file else 0
            
            # Get the corresponding directory for this txt_file in img_dir
            corresponding_dir = os.path.join(img_dir, file_base_name)
            
            # Check if the directory exists
            if os.path.exists(corresponding_dir):
                
                # Load the numpy predictions from preds.txt
                preds = np.loadtxt(os.path.join(corresponding_dir, 'preds.txt'))
                
                # Calculate the mean and standard deviation
                mean_pred = np.mean(preds)
                std_pred = np.std(preds)
                
                # Read the indices from the .mp4.txt file
                with open(os.path.join(txt_dir, txt_file), 'r') as f:
                    indices = f.readline().strip().split(',')
                    
                    for idx in indices:
                        idx_value = int(idx) + idx_offset
                        
                        # Check if index is within range for the specified criterion
                        for i in range(idx_value - frame_range, idx_value + frame_range + 1):
                            if (i >= 0 and i < len(preds) and 
                                abs(preds[i] - mean_pred) >= 2 * std_pred):
                                
                                img_name = f"img_{i:010}.jpg"
                                dataset.append([corresponding_dir, img_name, idx_value, aim_assist])
                                print(f"Appended: {corresponding_dir}, {img_name}, {idx_value}, {aim_assist}")
                                break
    
    # Write the dataset to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Directory Path', 'Filename', 'Index', 'Aim-Assist ON'])  # Writing the headers
        csvwriter.writerows(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center crop videos in a directory.")
    parser.add_argument("--txt_dir", required=True, help="Source directory containing the videos.")
    parser.add_argument("--img_dir", required=True, help="Destination directory to save the cropped videos.")
    parser.add_argument("--output_csv_file", type=str, default="combocheater.csv", help="Size of the center crop.")

    args = parser.parse_args()
    create_dataset_csv(args.txt_dir, args.img_dir, args.output_csv_file)
