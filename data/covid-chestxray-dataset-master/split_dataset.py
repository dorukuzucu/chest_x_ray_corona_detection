import os
import pandas as pd
from pathlib import Path
import shutil


root_dir = Path(__file__).parents[0]
csv_file = "metadata.csv"
metadata = pd.read_csv(os.path.join(root_dir,csv_file))

target_dir = os.path.join(Path(__file__).parents[2],"data","TRAIN")

print(metadata.head())
covid_count = 0
normal_count = 0

for row in metadata.iterrows():

    file_name = row[1]["filename"]
    label = row[1]["finding"]

    if label=="No Finding":
        image_path = os.path.join(root_dir, "images", file_name)
        target_path = os.path.join(target_dir, "normal")
        try:
            shutil.copy2(image_path,target_path)
        except:
            print("File skipped")
        normal_count+=1
    elif label=="Pneumonia/Viral/COVID-19":
        image_path = os.path.join(root_dir, "images", file_name)
        target_path = os.path.join(target_dir, "covid")
        try:
            shutil.copy2(image_path,target_path)
        except:
            print("File skipped")
        covid_count+=1

print("Total {} images are moved".format(covid_count+normal_count))
print("Copied Covid Images:{}".format(covid_count))
print("Copied Normal Images:{}".format(normal_count))

