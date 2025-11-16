from datasets import load_from_disk

# 1. Load the dataset you saved with save_to_disk
#    Replace 'hf_folder_name' with your dataset's folder path
hf_folder_name = "./hf_datasets/movielens-100k" 
ds_dict = load_from_disk(hf_folder_name)

# 2. Push the dataset to the Hub
#    This will create a new repository under your username.
#    Replace "my-awesome-dataset" with your desired repo name.
repo_name = "DukeNLPGroup/movielens-100k"
ds_dict.push_to_hub(repo_name)