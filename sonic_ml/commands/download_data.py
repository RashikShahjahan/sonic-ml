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
        
    The function performs the following steps:
    1. Checks if the dataset is already downloaded, if not downloads the specified dataset from Hugging Face using the datasets library
    2. Saves the entire dataset to disk in the specified output directory
    3. Returns the path to the saved dataset
    
    Example:
        >>> output_path = download_dataset(
        ...     dataset_name='wikitext',
        ...     dataset_data_dir='data/raw',
        ...     output_dir='data/processed'
        ... )
    """
    if os.path.exists(output_dir):
        print(f"Dataset already downloaded at {output_dir}")
        return output_dir
    
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









