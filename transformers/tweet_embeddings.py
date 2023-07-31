"""
This script is used to the XLNet and RoBERTa model with different tweet embedddings generation methods
Run using command: python tweet_embeddings.py model_type combine_type
model_type can be: xlnet, roberta
combine_type can be: l1, wl4, l4cnn, l8cnn
"""

from dataset import (
    load_dataset,
    tokenize_dataset,
    load_and_tokenize_dataset,
    get_obervation_dataset,
)
import sys
import transformers
from transformers import RobertaModel, AutoModelForSequenceClassification
import torch
from typing import NamedTuple
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import copy
from torch import cuda
import pickle
from pprint import pprint
import pandas as pd

np.random.seed(423)

VALIDATION_RESULTS = []
TRAIN_RESULTS = []

class ModelConfig(NamedTuple):
    tokenizer_model: str
    max_length: int
    nn_model: str
    device: str
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    start_epoch: int
    learning_rate: float
    dataset_type: str
    force_reload_dataset: bool

    def __repr__(self) -> str:
        out = "Best Transformer Config:\n"
        for k, v in zip(self._fields, self):
            out += f"- {k:<16} = {v}\n"
        return out

# XLNet, last hidden layer + dense 
class XLNetClass_l1(torch.nn.Module):
    def __init__(self):
        super(XLNetClass_l1, self).__init__()
        self.l1 = transformers.XLNetModel.from_pretrained(
            'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary', return_dict=True, output_hidden_states=True)
        """
        for name, param in self.l1.named_parameters():
          if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name or "layer.7" in name or "layer.6" in name:
            param.requires_grad=True
          else:
            param.requires_grad=False
        """
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1)
        self.sigmoid= torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.pool_hidden_state(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
    
    # pool the last hidden layer
    def pool_hidden_state(self, output):
        last_hidden_state = output[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

# XLNet, weighted average last 4 layers + dense
class XLNetClass_wl4(torch.nn.Module):
    def __init__(self):
        super(XLNetClass_wl4, self).__init__()
        self.l1 = transformers.XLNetModel.from_pretrained(
            'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary', return_dict=True, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        self.layer_weights = torch.nn.Parameter(torch.tensor([1] * 4, dtype=torch.float))
    
    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.weighted_average_last_four_layers(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    # weighted average last 4 hidden states
    def weighted_average_last_four_layers(self, output):
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 16 * 45 * 768)
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(last_four.size())
        weighted_average = (weight_factor * last_four).sum(dim=0) / self.layer_weights.sum()
        sentence_last_four = torch.mean(weighted_average, 1)
        return sentence_last_four

# XLNet, last 4 hidden layers + cnn
class XLNetClass_l4_cnn(torch.nn.Module):
    def __init__(self):
        super(XLNetClass_l4_cnn, self).__init__()
        self.l1 = transformers.XLNetModel.from_pretrained(
            'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary', return_dict=True, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(256, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        self.conv2d = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.relu = torch.nn.ReLU()
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.cnn2d_last_four_layers(output)
        output = self.conv2d(output)
        output = self.relu(output)
        output = self.maxpool2d(output)
        output = output.squeeze(dim=3)
        output = output.squeeze(dim=2)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def cnn2d_last_four_layers(self, output):
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 16 * 45 * 768)
        last_four = last_four.permute(1, 0, 2, 3) # (32, 4, 45 ,768)
        return last_four

# XLNet, last 4 and intermediate 4 hidden layers + cnn
class XLNetClass_l8_cnn(torch.nn.Module):
    def __init__(self):
        super(XLNetClass_l8_cnn, self).__init__()
        self.l1 = transformers.XLNetModel.from_pretrained(
            'Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary', return_dict=True, output_hidden_states=True)
        
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(512, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        
        self.conv2d = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.conv2dm = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.relu = torch.nn.ReLU()
        self.relum = torch.nn.ReLU()
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling
        self.maxpool2dm = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling
        

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

        # cnn 2d only last four
        last = self.cnn2d_four_layers(output, 9, 13)
        last = self.conv2d(last)
        last = self.relu(last)
        last = self.maxpool2d(last)
        last = last.squeeze(dim=3)
        last = last.squeeze(dim=2)
        
        # cnn 2d only middle four
        mid = self.cnn2d_four_layers(output, 5, 9)
        mid = self.conv2dm(mid)
        mid = self.relum(mid)
        mid = self.maxpool2dm(mid)
        mid = mid.squeeze(dim=3)
        mid = mid.squeeze(dim=2)
    
        # concate
        output = torch.cat((mid, last), dim=1)
        
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def cnn2d_four_layers(self, output, start, end):
        hidden_states = output["hidden_states"]
        four_layers = torch.stack(hidden_states)[start:end, :, :, :] # (4 * 16 * 45 * 768)
        four_layers = four_layers.permute(1, 0, 2, 3)
        return four_layers

    def cnn2d_last_four_layers(self, output):
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 16 * 45 * 768)
        last_four = last_four.permute(1, 0, 2, 3)
        # print("last_four: ", last_four.shape) # (32, 4, 45 ,768)
        return last_four

    def cnn2d_middle_four_layers(self, output):
        hidden_states = output["hidden_states"]
        middle_four = torch.stack(hidden_states)[5:9, :, :, :] # (4 * 16 * 45 * 768)
        middle_four = middle_four.permute(1, 0, 2, 3)
        # print("middle_four: ", middle_four.shape) # (32, 4, 45 ,768)
        return middle_four



    def concat_last_four(self, output): # TODO: concate last 4
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 32 * 45 * 768)
        sentence_last_four = torch.mean(last_four, 2) # (4 * 32 * 768)
        # print("sentence_last_four", sentence_last_four.shape)
        #cat_last_four = torch.tensor()
        cat_last_four = torch.cat((torch.tensor(sentence_last_four[0]),
                    torch.tensor(sentence_last_four[1]),
                    torch.tensor(sentence_last_four[2]),
                   torch.tensor(sentence_last_four[3])), -1)
        # print("cat_last_four", cat_last_four.shape) # (32, 3072)
        return cat_last_four
        
# RoBERTa, last hidden layer + dense 
class RoBERTaClass_l1(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass_l1, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1)
        self.sigmoid= torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        # output = self.pool_hidden_state(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

# RoBERTa, weighted average last 4 layers + dense
class RoBERTaClass_wl4(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass_wl4, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        self.layer_weights = torch.nn.Parameter(torch.tensor([1] * 4, dtype=torch.float))
    
    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.weighted_average_last_four_layers(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    # weighted average last 4 hidden states
    def weighted_average_last_four_layers(self, output):
        # print("layers shape", len(output[2]))
        hidden_states = output[2]
        last_four = torch.stack(hidden_states)[8:12, :, :, :] # (4 * 16 * 45 * 768)
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(last_four.size())
        weighted_average = (weight_factor * last_four).sum(dim=0) / self.layer_weights.sum()
        sentence_last_four = torch.mean(weighted_average, 1)
        return sentence_last_four

# RoBERTa, last 4 hidden layers + cnn
class RoBERTaClass_l4_cnn(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass_l4_cnn, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(256, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        self.conv2d = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.relu = torch.nn.ReLU()
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.cnn2d_last_four_layers(output)
        output = self.conv2d(output)
        output = self.relu(output)
        output = self.maxpool2d(output)
        output = output.squeeze(dim=3)
        output = output.squeeze(dim=2)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def cnn2d_last_four_layers(self, output):
        hidden_states = output[2]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 16 * 45 * 768)
        last_four = last_four.permute(1, 0, 2, 3) # (32, 4, 45 ,768)
        return last_four

# RoBERTa, last 4 and intermediate 4 hidden layers + cnn
class RoBERTaClass_l8_cnn(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass_l8_cnn, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(512, 1) # (3072 - concate, 768 - linear, 256 - cnn, 512 - cnn_cat)
        self.sigmoid= torch.nn.Sigmoid()
        
        self.conv2d = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.conv2dm = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.relu = torch.nn.ReLU()
        self.relum = torch.nn.ReLU()
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling
        self.maxpool2dm = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling
        

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

        # cnn 2d only last four
        last = self.cnn2d_four_layers(output, 9, 13)
        last = self.conv2d(last)
        last = self.relu(last)
        last = self.maxpool2d(last)
        last = last.squeeze(dim=3)
        last = last.squeeze(dim=2)
        
        # cnn 2d only middle four
        mid = self.cnn2d_four_layers(output, 5, 9)
        mid = self.conv2dm(mid)
        mid = self.relum(mid)
        mid = self.maxpool2dm(mid)
        mid = mid.squeeze(dim=3)
        mid = mid.squeeze(dim=2)
    
        # concate
        output = torch.cat((mid, last), dim=1)
        
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def cnn2d_four_layers(self, output, start, end):
        hidden_states = output[2]
        four_layers = torch.stack(hidden_states)[start:end, :, :, :] # (4 * 16 * 45 * 768)
        four_layers = four_layers.permute(1, 0, 2, 3)
        return four_layers

    def cnn2d_last_four_layers(self, output):
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 16 * 45 * 768)
        last_four = last_four.permute(1, 0, 2, 3)
        # print("last_four: ", last_four.shape) # (32, 4, 45 ,768)
        return last_four

    def cnn2d_middle_four_layers(self, output):
        hidden_states = output["hidden_states"]
        middle_four = torch.stack(hidden_states)[5:9, :, :, :] # (4 * 16 * 45 * 768)
        middle_four = middle_four.permute(1, 0, 2, 3)
        # print("middle_four: ", middle_four.shape) # (32, 4, 45 ,768)
        return middle_four

    def concat_last_four(self, output):
        hidden_states = output["hidden_states"]
        last_four = torch.stack(hidden_states)[9:, :, :, :] # (4 * 32 * 45 * 768)
        sentence_last_four = torch.mean(last_four, 2) # (4 * 32 * 768)
        # print("sentence_last_four", sentence_last_four.shape)
        #cat_last_four = torch.tensor()
        cat_last_four = torch.cat((torch.tensor(sentence_last_four[0]),
                    torch.tensor(sentence_last_four[1]),
                    torch.tensor(sentence_last_four[2]),
                   torch.tensor(sentence_last_four[3])), -1)
        # print("cat_last_four", cat_last_four.shape) # (32, 3072)
        return cat_last_four
   
class RoBERTaTwitterEN(torch.nn.Module):
    def __init__(self):
        print("using RoBERTaTwitterEN")
        super(RoBERTaTwitterEN, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/roberta-base-tweet-sentiment-en",
            output_hidden_states=True,
            return_dict=True,
        )

        # self.l2 = torch.nn.Linear(3, 3)
        # self.l3 = torch.nn.RReLU()
        self.l4 = torch.nn.Linear(3, 1)
        self.l5 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # output = self.l2(output[0])
        # output = self.l3(output)
        output = self.l4(output[0])
        output = self.l5(output)
        return output

def train_model(
    model,
    model_config: ModelConfig,
    train_data,
    val_data,
    *,
    store_path_tmpl=None,
):
    """
    Train the model
    """

    print(
        f"Training model for {model_config.epochs} epochs starting at {model_config.start_epoch}"
    )

    loss_fn = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    device = model_config.device

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    for epoch in range(model_config.start_epoch, model_config.epochs):
        model.train()
        with tqdm(
            train_data,
            unit="batch",
            # mininterval=5,
            # maxinterval=10,
            miniters=50,
        ) as bar:
            bar.set_description(f"Epoch {epoch}")

            correct_cnt = 0
            cnt = 0
            for i, batch in enumerate(bar):
                # forward pass
                y_batch = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                y_pred = model(
                    batch["input_ids"].to(device, dtype=torch.long),
                    batch["attention_mask"].to(device, dtype=torch.long),
                    batch["token_type_ids"].to(device, dtype=torch.long),
                    # batch["irony"].to(device, dtype=torch.float32),
                )
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress

                cnt += len(y_batch)
                correct_cnt += int((y_pred.round() == y_batch).int().sum())

                if i % 50 == 0:
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=f"{float(loss):.3f}",
                        acc=f"{float(acc):.3f}",
                        total_acc=f"{correct_cnt/cnt:.3f}",
                    )

            acc = correct_cnt / cnt
            print(f"Epoch {epoch} training accuracy: {acc:.3f}")

        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            correct_cnt = 0
            cnt = 0
            for val_batch in tqdm(val_data, desc="Validation"):
                y_val = (
                    val_batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                )
                y_pred = model(
                    val_batch["input_ids"].to(device, dtype=torch.long),
                    val_batch["attention_mask"].to(device, dtype=torch.long),
                    val_batch["token_type_ids"].to(device, dtype=torch.long),
                    # batch["irony"].to(device, dtype=torch.float32),
                )
                cnt += len(y_val)
                correct_cnt += int((y_pred.round() == y_val).int().sum())

        acc = correct_cnt / cnt
        print(f"Epoch {epoch} validation accuracy: {acc:.3f}\n")

        if store_path_tmpl is not None:
            save_model(model, store_path_tmpl.format(epoch))

        # if acc > best_acc:
        #     best_acc = acc
        #     best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    # model.load_state_dict(best_weights)
    return best_acc

def eval_model(model, model_config: ModelConfig, train_data, val_data):
    """
    Evaluate model on train and validation data
    """

    device = model_config.device
    model.eval()
    with torch.no_grad():
        correct_cnt = 0
        cnt = 0
        for batch in tqdm(val_data):
            y = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y)
            correct_cnt += int((y_pred.round() == y).int().sum())

        acc = correct_cnt / cnt
        print(f"validation accuracy: {acc:.3f}")

        correct_cnt = 0
        cnt = 0
        for batch in tqdm(train_data):
            y = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y)
            correct_cnt += int((y_pred.round() == y).int().sum())

        acc = correct_cnt / cnt
        print(f"training accuracy: {acc:.3f}")

def observe_model(model, model_config: ModelConfig):
    dataset = get_obervation_dataset(model_config, frac=1, train_size=0.85)

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
    )

    correct_output_file = "correct.csv"
    incorrect_output_file = "incorrect.csv"

    correct_file = open(correct_output_file, "w")
    incorrect_file = open(incorrect_output_file, "w")
    model.eval()
    device = model_config.device
    with torch.no_grad():
        correct_cnt = 0
        cnt = 0
        for val_batch in tqdm(validation_loader, desc="Validation"):
            y_val = val_batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
            y_pred = model(
                val_batch["input_ids"].to(device, dtype=torch.long),
                val_batch["attention_mask"].to(device, dtype=torch.long),
                val_batch["token_type_ids"].to(device, dtype=torch.long),
            )
            cnt += len(y_val)
            correct = (y_pred.round() == y_val).int().cpu()
            correct_cnt += int((y_pred.round() == y_val).int().sum())

            for i, c in enumerate(correct):
                if c == 1:
                    print(
                        f"{int(val_batch['label'][i])}: {val_batch['tweet'][i]}",
                        file=correct_file,
                    )
                elif c == 0:
                    print(
                        f"{int(val_batch['label'][i])}: {val_batch['tweet'][i]}",
                        file=incorrect_file,
                    )
                else:
                    raise ValueError("incorrect value for correct")

        print(f"correct: {correct_cnt/cnt:.3f}")
        correct_file.close()
        incorrect_file.close()

def generate_predictions(model, model_config: ModelConfig, test_data, output_file):
    """
    Generate predictions for test data that can be submitted to Kaggle
    """
    model.eval()
    device = model_config.device

    output = pd.DataFrame(columns=["Id", "Prediction"])

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating predictions"):
            ids = batch["id"].reshape(-1, 1)
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            ).round()

            # move tensors to cpu
            ids = ids.cpu()
            y_pred = y_pred.cpu()

            df = pd.DataFrame({"Id": ids.squeeze(), "Prediction": y_pred.squeeze()})
            df["Prediction"] = df["Prediction"].apply(lambda x: 1 if x == 1 else -1)

            output = pd.concat([output, df])

    output.to_csv(output_file, index=False, sep=",", header=True)
    print(f"generated predictions ({output_file=})")

def generate_val_predictions(model, model_config: ModelConfig, val_data, output_file):
    """
    Generate predictions for test data that can be submitted to Kaggle
    """
    model.eval()
    device = model_config.device

    output = pd.DataFrame(columns=["Prediction", "Label", "Tweet"])

    with torch.no_grad():
        for batch in tqdm(val_data, desc="Generating validation predictions"):
            labels = batch["label"].reshape(-1, 1)
            tweets = batch["tweet"]
            y_pred = model(
                batch["input_ids"].to(device, dtype=torch.long),
                batch["attention_mask"].to(device, dtype=torch.long),
                batch["token_type_ids"].to(device, dtype=torch.long),
            ).round()

            # move tensors to cpu
            labels = labels.cpu()
            y_pred = y_pred.cpu()

            df = pd.DataFrame(
                {
                    "Prediction": y_pred.squeeze(),
                    "Label": labels.squeeze(),
                    "Tweet": tweets,
                }
            )
            df["Prediction"] = df["Prediction"].apply(lambda x: 1 if x == 1 else -1)
            df["Label"] = df["Label"].apply(lambda x: 1 if x == 1 else -1)

            output = pd.concat([output, df])

    output.to_csv(output_file, index=False, sep="\t", header=True)
    print(f"generated validation predictions ({output_file=})")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def main():
   
 
    # total arguments
    num_arg = len(sys.argv)
    print("Total arguments passed:", num_arg)
    if num_arg != 3:
        print("Invalid arguments. Should be python xlnet_roberta.py model_type combine_type")
        return
    
    model_type = sys.argv[1]
    combine_type = sys.argv[2]

    model_config = None
    model = None

    if model_type == "xlnet":
        
        model_config = ModelConfig(
            tokenizer_model="xlnet-base-cased",
            max_length=45,
            nn_model="Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary",
            device="cuda" if cuda.is_available() else "cpu",
            train_batch_size=32,
            valid_batch_size=32,
            epochs=4,
            start_epoch=0,
            learning_rate=1e-05,
            dataset_type="combined",
            force_reload_dataset=False,
        )
        
        if combine_type == "l1":
            model = XLNetClass_l1()
        elif combine_type == "wl4":
            model = XLNetClass_wl4()
        elif combine_type == "l4cnn":
            model = XLNetClass_l4_cnn()
        elif combine_type == "l8cnn":
            model = XLNetClass_l8_cnn()
        else:
            print("Invalid combine type. Should be l1, wl4, l4cnn, or l8cnn")
            return
        
    elif model_type == 'roberta':

        model_config = ModelConfig(
            tokenizer_model="roberta-base",
            max_length=45,
            nn_model="roberta-base",
            device="cuda" if cuda.is_available() else "cpu",
            train_batch_size=32,
            valid_batch_size=32,
            epochs=4,
            start_epoch=0,
            learning_rate=1e-05,
            dataset_type="combined",
            force_reload_dataset=False,
        )
    
        if combine_type == "l1":
            model = RoBERTaClass_l1()
        elif combine_type == "wl4":
            model = RoBERTaClass_wl4()
        elif combine_type == "l4cnn":
            model = RoBERTaClass_l4_cnn()
        elif combine_type == "l8cnn":
            model = RoBERTaClass_l8_cnn()
        else:
            print("Invalid combine type. Should be l1, wl4, l4cnn, or l8cnn")
            return
    else:
        print("Invalid model type. Should be xlnet or roberta.")
        return
    
    print(model_config)
    model.to(model_config.device)

    dataset = load_and_tokenize_dataset(
        model_config,
        frac=1,
        train_size=0.50,
        force_reload=model_config.force_reload_dataset,
        include_tweet=True,
    )
    training_loader = DataLoader(
        dataset["train"],
        batch_size=model_config.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    train_model(model, model_config, training_loader, validation_loader)
    eval_model(model, model_config, training_loader, validation_loader)
    save_model(model, model_type + "_" + combine_type + "_4_epoch.pkl" )
    generate_predictions(
        model,
        model_config,
        test_loader,
        model_type + "_" + combine_type + "_4_epoch_test_results.csv",
    )
    

            
if __name__ == "__main__":
    main()
