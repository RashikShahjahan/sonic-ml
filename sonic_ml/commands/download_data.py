import os
from datasets import load_dataset

def download_dataset(dataset_name: str, dataset_data_dir: str, output_dir: str) -> str:
    """Download a dataset from Hugging Face and save it locally.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., 'wikipedia', 'wikitext')
        dataset_data_dir (str): Directory containing additional data files required by the dataset
        output_dir (str): Local directory path where the dataset will be saved
    
    Returns:
        str: Path to the directory where the dataset was saved
    """
    if os.path.exists(output_dir):
        print(f"Dataset already downloaded at {output_dir}")
        return output_dir
    
    print(dataset_name, dataset_data_dir)
    dataset = load_dataset(dataset_name, data_dir=dataset_data_dir)   
    
    dataset.save_to_disk(output_dir)
    return output_dir



def download_workflow(dataset_name: str, dataset_data_dir: str, output_dir: str):
    """Workflow for downloading and saving a dataset from Hugging Face Hub locally.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., 'wikipedia', 'wikitext')
        dataset_data_dir (str): Directory containing additional data files required by the dataset
        output_dir (str): Local directory path where the dataset will be saved

    """
    download_dataset(dataset_name=dataset_name, dataset_data_dir=dataset_data_dir, output_dir=output_dir)









