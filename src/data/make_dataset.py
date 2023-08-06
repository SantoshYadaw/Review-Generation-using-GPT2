import os
import re
import logging
from tqdm.auto import tqdm

tqdm.pandas()

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def clean(review: str):
    """Perform simple text processing such as converting to lowercase, remove numbers and stop words.

    Args:
        review (str): Text

    Returns:
        review (str): Cleaned text
    """
    # Convert to lower case
    review = review.lower()
    # Remove any numbers
    review = re.sub("[^a-z A-Z 0-9-]+", "", review)
    # Remove any stopwords in english
    review = " ".join(
        [word for word in review.split() if word not in stopwords.words("english")]
    )

    return review


def prepare_dataset_gpt(raw_data_path: str, logging):
    """Prepare dataset for training gpt2

    Args:
        raw_data_path (str): Raw data path
        logging (_type_): Logger
    """

    # Load raw data
    logging.info(f"Loading raw data from: {raw_data_path}")
    data = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    data = data.T.reset_index().T.reset_index(drop=True).rename(columns={0: "text"})

    # Remove duplicates
    logging.info(f"Preprocessing data")
    data = data[~data["text"].duplicated()]

    # Clean up text
    data["text"] = data["text"].progress_apply(clean)
    data.to_csv(INTERIM_DATA_PATH, index=False)

    # Split randomly
    data_train, data_val = train_test_split(
        data, train_size=TRAIN_SIZE, random_state=SEED
    )
    len_data_train = len(data_train)
    len_data_val = len(data_val)
    logging.info(
        f"There are {len_data_train} samples for training and {len_data_val} for validation"
    )

    # Save
    logging.info(f"Saving data")
    data_train["split"] = "train"
    data_val["split"] = "val"

    combined_data = pd.concat([data_train, data_val])
    combined_data.to_csv(SPLIT_DATA_PATH, index=False)
    logging.info(f"Prepearation of dataset complete. Data saved to {SPLIT_DATA_PATH}")


if __name__ == "__main__":

    # Setup logger
    logging.basicConfig(level=logging.INFO)

    # Constants
    HOME_PATH = os.getcwd()
    logging.info(f"HOME_PATH: {HOME_PATH}")

    DATA_PATH = os.path.join(HOME_PATH, "data", "raw", "task2.csv")
    logging.info(f"DATA_PATH: {DATA_PATH}")

    INTERIM_DATA_PATH = os.path.join(HOME_PATH, "data", "interim", "interim_data.csv")
    logging.info(f"INTERIM_DATA_PATH: {INTERIM_DATA_PATH}")

    SPLIT_DATA_PATH = os.path.join(HOME_PATH, "data", "processed", "split_data.csv")
    logging.info(f"SPLIT_DATA_PATH: {SPLIT_DATA_PATH}")

    # Data split constants
    TRAIN_SIZE = 0.9
    SEED = 2023

    prepare_dataset_gpt(DATA_PATH, logging)
