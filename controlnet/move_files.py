from pathlib import Path
from tqdm import tqdm

from utils import root, original_path, get_and_create_path


new_path = get_and_create_path("seg")
print(new_path)
print(new_path.exists())

for folder in original_path.iterdir():
    for image in tqdm(list(folder.iterdir()), desc=folder.name):
        if "aug" in image.name:
            new_name = image.name[:-7] + "seg.jpg"
            image.rename(new_path / folder.name / new_name)
