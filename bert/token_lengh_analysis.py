"""
This script is used to analyze the length of the tweets in the dataset.

It creates a histogram of the length of the tweets in the dataset.
"""


from main import ModelConfig
from dataset import (
    load_and_tokenize_dataset,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

model_config = ModelConfig(
    tokenizer_model="prajjwal1/bert-mini",
    max_length=200,
    nn_model="prajjwal1/bert-mini",
    device="cpu",
    train_batch_size=32,
    valid_batch_size=32,
    epochs=4,
    start_epoch=0,
    learning_rate=1e-05,
    dataset_type="combined",
    force_reload_dataset=True,
    weight_store_template="",
)


dataset = load_and_tokenize_dataset(
    model_config,
    frac=1,
    train_size=1,
    force_reload=model_config.force_reload_dataset,
    include_tweet=True,
)


dataset = dataset["train"]

tweet_lenghts = []
token_cnts = []
for data_point in tqdm(dataset):
    tweet_lenghts.append(len(data_point["tweet"].split(" ")))
    token_cnts.append(data_point["attention_mask"].sum().item())


print(
    "Percentage of tweets with length > 45: {:.4%}".format(
        sum(np.array(token_cnts) <= 45) / len(token_cnts)
    )
)

sns.set()
plt.rcParams["figure.figsize"] = (15, 5)
f = plt.figure()
f.add_subplot()
sns.distplot(tweet_lenghts)

plt.ylabel("density", fontsize=16)
plt.xlabel("token counts", fontsize=16)
plt.xlim([0, 60])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.title("token counts", fontsize=20)

plt.tight_layout()
plt.savefig("token_counts.pdf")
# plt.show()
