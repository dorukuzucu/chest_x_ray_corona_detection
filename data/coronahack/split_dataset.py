import os
import pandas as pd
from pathlib import Path
import shutil


root_dir = os.path.join(Path(__file__).parents[0])
csv_file = "metadata.csv"
metadata = pd.read_csv(os.path.join(root_dir,csv_file))

target_dir = Path(__file__).parents[2]

print(metadata.head())
covid_count = 0
normal_count = 0
for row in metadata.iterrows():
    # print(row[1]["Dataset_type"],row[1]["X_ray_image_name"],row[1]["Label"],row[1]["Label_2_Virus_category"])

    dataset = row[1]["Dataset_type"]
    file_name = row[1]["X_ray_image_name"]
    label = row[1]["Label"]
    label_2 = row[1]["Label_2_Virus_category"]

    if label=="Normal":
        image_path = os.path.join(root_dir, dataset, file_name)
        target_path = os.path.join(target_dir, dataset,"normal")
        shutil.copy2(image_path,target_path)
        normal_count+=1
    elif label!="Normal" and label_2=="COVID-19":
        image_path = os.path.join(root_dir, "coronahack", dataset, file_name)
        target_path = os.path.join(target_dir, dataset, "covid")
        shutil.copy2(image_path, target_path)
        covid_count+=1

print("Total {} images are moved".format(covid_count+normal_count))
print("Copied Covid Images:{}".format(covid_count))
print("Copied Normal Images:{}".format(normal_count))


# X_ray_image_name, Label, Dataset_type, Label_2_Virus_category, Label_1_Virus_category