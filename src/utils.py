import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_gap_feature_importance(
    input: pd.DataFrame, prefix: str, item_labels: list, max_val=200
):

    # We slightly format the topics and labels for them to fit on the plot.
    topic_labels = [x.replace(f"{prefix}: ", "") for x in input.columns]
    labels = item_labels

    # We clip large outliers to makes activations more visible.
    input = np.clip(input, a_min=None, a_max=max_val)

    plt.figure(figsize=(10, 10), dpi=200)

    plt.imshow(input.T)

    plt.yticks(
        range(len(topic_labels)),
        labels=topic_labels,
        ha="right",
        size=12,
    )
    plt.xticks(range(len(labels)), labels=labels, size=12, rotation=50, ha="right")

    plt.colorbar().set_label(label="Topic activations", size=13)
    plt.ylabel("Latent topics", size=14)
    plt.xlabel("Items", size=14)
    plt.tight_layout()
    plt.show()


def preprocess_text(
    input_text: str = None,
    pattern: re.Pattern = r"[^\w\s\u00C0-\u017F]+|\xa0",
) -> str:
    """
    Normalize text by removing specific characters
    """
    ## NOTE: This is just basic preprocessing.
    output_text = re.sub(
        r"&quot;", " ", str(input_text).lower()
    )  ## "&quot;" seems to be an encoding issue that appears in severl records
    output_text = re.sub(pattern, " ", output_text)
    output_text = re.sub("\s\s+", " ", output_text)
    output_text = output_text.strip()
    return output_text
