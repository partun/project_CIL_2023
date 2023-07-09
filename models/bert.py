import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from tqdm import tqdm

device = "cuda" if cuda.is_available() else "cpu"

print(f"Using {device} device")


tweets_neg = pd.read_csv(
    "../twitter-datasets/train_neg_full.txt",
    sep="\t\t",
    lineterminator="\n",
    encoding="utf8",
    names=["tweet"],
)
tweets_pos = pd.read_csv(
    "../twitter-datasets/train_pos_full.txt",
    sep="\t",
    lineterminator="\n",
    encoding="utf8",
    names=["tweet"],
)

tweets_neg["label"] = "pos"
tweets_pos["label"] = "neg"
tweets = pd.concat([tweets_neg, tweets_pos])

tweets = tweets.sample(frac=1).reset_index(drop=True)

print(f"Total tweets: {len(tweets)}")

tweets.head()


# Sections of config

# Defining some key variables that will be used later on in the training
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.tweets = dataframe.tweet
        self.targets = self.data.label
        self.max_len = 140

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        tweet = str(self.tweets[index])
        tweet = " ".join(tweet.split())

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        match self.targets[index]:
            case "neg":
                target = torch.tensor([0, 1], dtype=torch.float)
            case "pos":
                target = torch.tensor([1, 0], dtype=torch.float)
            case invalid:
                raise ValueError(f"Invalid label {invalid}")

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": target,
        }


# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset = tweets.sample(frac=train_size, random_state=200)
test_dataset = tweets.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(tweets.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer)
testing_set = CustomDataset(test_dataset, tokenizer)


train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(
            "prajjwal1/bert-mini", return_dict=False
        )
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(256, 2)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = BERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for i, data in enumerate(training_loader, 0):
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i % 5000 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation(epoch, loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


for epoch in tqdm(range(EPOCHS)):
    train(epoch)

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch, training_loader)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    print(f"Accuracy Score (Train) = {accuracy}")

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch, testing_loader)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    print(f"Accuracy Score (Test) = {accuracy}")
