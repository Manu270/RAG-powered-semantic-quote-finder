
from datasets import load_dataset
import pandas as pd

def load_and_prepare_quotes():
    dataset = load_dataset("Abirate/english_quotes", split="train")
    df = pd.DataFrame(dataset)
    df.dropna(subset=["quote", "author", "tags"], inplace=True)
    df["text"] = df["quote"] + " - " + df["author"] + " [" + df["tags"].apply(lambda x: ', '.join(x)) + "]"
    return df
