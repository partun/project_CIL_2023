import dataset
from dataset import load_dataset, tokenize_dataset, load_and_tokenize_dataset
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

np.random.seed(1)

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
    learning_rate: float
    dataset_type: str
    force_reload_dataset: bool

    def __repr__(self) -> str:
        out = "Best Transformer Config:\n"
        for k, v in zip(self._fields, self):
            out += f"- {k:<16} = {v}\n"
        return out


class RoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base", return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
        self.l4 = torch.nn.Sigmoid()
        self.layer_weights = torch.nn.Parameter(torch.tensor([1] * 4, dtype=torch.float))

        self.conv1d = torch.nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding='valid', stride=1)
        self.conv2d = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.conv2dm = torch.nn.Conv2d(in_channels=4, out_channels=256, kernel_size=(3, 768), stride=1)
        self.relu = torch.nn.ReLU()
        self.relum = torch.nn.ReLU()
        self.maxpool1d = torch.nn.MaxPool1d(kernel_size=45-3+1)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling
        self.maxpool2dm = torch.nn.MaxPool2d(kernel_size=(43, 1)) # global max pooling

    def forward(self, ids, mask, token_type_ids):
        _, output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # output = self.weighted_average_last_four_layers(output)

        """
        last = self.cnn2d_last_four_layers(output)
        last = self.conv2d(last)
        # print("after_con2d: ", output.shape) # (32, 256, 43, 1) ?
        last = self.relu(last)
        # print("after relu: ", output.shape)
        last = self.maxpool2d(last)
        # print("after maxpool2d: ", output.shape)
        last = last.squeeze(dim=3)
        # print("after flatten: ", output.shape)
        output = last.squeeze(dim=2)
        # print("after flatten: ", output.shape) # (32, 256)
        

        # cnn 2d only last four
        last = self.cnn2d_last_four_layers(output)
        last = self.conv2d(last)
        # print("after_con2d: ", output.shape) # (32, 256, 43, 1) ?
        last = self.relu(last)
        # print("after relu: ", output.shape)
        last = self.maxpool2d(last)
        # print("after maxpool2d: ", output.shape)
        last = last.squeeze(dim=3)
        # print("after flatten: ", output.shape)
        last = last.squeeze(dim=2)
        # print("after flatten: ", output.shape) # (32, 256)
        

        # cnn 2d only last four
        mid = self.cnn2d_middle_four_layers(output)
        mid = self.conv2dm(mid)
        # print("after_con2d: ", output.shape) # (32, 256, 43, 1) ?
        mid = self.relum(mid)
        # print("after relu: ", output.shape)
        mid = self.maxpool2dm(mid)
        # print("after maxpool2d: ", output.shape)
        mid = mid.squeeze(dim=3)
        # print("after flatten: ", output.shape)
        mid = mid.squeeze(dim=2)
        # print("after flatten: ", output.shape) # (32, 256)
        

        # concate
        output = torch.cat((mid, last), dim=1)
        # print("concate_shape", output.shape)
        """


        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output

    def weighted_average_last_four_layers(self, output):
        # print("layers shape", len(output[2]))
        hidden_states = output[2]
        last_four = torch.stack(hidden_states)[8:12, :, :, :] # (4 * 16 * 45 * 768)
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(last_four.size())
        weighted_average = (weight_factor * last_four).sum(dim=0) / self.layer_weights.sum()
        sentence_last_four = torch.mean(weighted_average, 1)
        return sentence_last_four

    def cnn2d_last_four_layers(self, output):
        hidden_states = output[2]
        last_four = torch.stack(hidden_states)[8:12, :, :, :] # (4 * 16 * 45 * 768)
        last_four = last_four.permute(1, 0, 2, 3)
        # print("last_four: ", last_four.shape) # (32, 4, 45 ,768)
        return last_four

    def cnn2d_middle_four_layers(self, output):
        hidden_states = output[2]
        middle_four = torch.stack(hidden_states)[4:8, :, :, :] # (4 * 16 * 45 * 768)
        middle_four = middle_four.permute(1, 0, 2, 3)
        # print("middle_four: ", middle_four.shape) # (32, 4, 45 ,768)
        return middle_four

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
   
def train_model(model, model_config: ModelConfig, train_data, val_data):
    """
    Train the model
    """

    loss_fn = torch.nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    device = model_config.device

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    for epoch in range(model_config.epochs):
        model.train()
        with tqdm(
            train_data,
            unit="batch",
            mininterval=0,
            miniters=200,
        ) as bar:
            bar.set_description(f"Epoch {epoch}")

            correct_cnt = 0
            cnt = 0
            for i, batch in enumerate(bar):
                # take a batch

                # forward pass
                y_batch = batch["label"].to(device, dtype=torch.float32).reshape(-1, 1)
                y_pred = model(
                    batch["input_ids"].to(device, dtype=torch.long),
                    batch["attention_mask"].to(device, dtype=torch.long),
                    batch["token_type_ids"].to(device, dtype=torch.long),
                )
                loss = loss_fn(y_pred, y_batch)
                cnt += len(y_batch)
                correct_cnt += int((y_pred.round() == y_batch).int().sum())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress

                if i % 200 == 0:
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=f"{float(loss):.4f}",
                        acc=f"{float(acc):.4f}",
                        total_acc=f"{correct_cnt/cnt:.4f}",
                    )

            acc = correct_cnt / cnt
            print(f"Epoch {epoch} training accuracy: {acc:.4f}")

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
                )
                cnt += len(y_val)
                correct_cnt += int((y_pred.round() == y_val).int().sum())

        acc = correct_cnt / cnt
        print(f"Epoch {epoch} validation accuracy: {acc:.4f}")
        VALIDATION_RESULTS.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
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
        print(f"validation accuracy: {acc:.4f}")
        VALIDATION_RESULTS.append(acc)
        
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
        print(f"training accuracy: {acc:.4f}")
        TRAIN_RESULTS.append(acc)

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

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def main():

    """
    model_config = ModelConfig(
            tokenizer_model="xlnet-base-cased",
            max_length=45,
            nn_model="Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary",
            device="cuda" if cuda.is_available() else "cpu",
            train_batch_size=32,
            valid_batch_size=32,
            epochs=3,
            learning_rate=1e-05,
            dataset_type="combined",
            force_reload_dataset=False,
    )

    model = XLNetClass_l8_cnn()
    """

    model_config = ModelConfig(
            tokenizer_model="roberta-base",
            max_length=45,
            nn_model="roberta-base",
            device="cuda" if cuda.is_available() else "cpu",
            # device = "cpu",
            train_batch_size=32,
            valid_batch_size=32,
            epochs=1,
            learning_rate=1e-05,
            dataset_type="combined",
            force_reload_dataset=False,
    )
    model = RoBERTaClass_l8_cnn()
    

    model.to(model_config.device)

    dataset = load_and_tokenize_dataset(
        model_config,
        frac=1,
        train_size=0.8,
        force_reload=model_config.force_reload_dataset
    )

    training_loader = DataLoader(
        dataset["train"],
        batch_size=model_config.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=model_config.valid_batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=model_config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # load_model(model, "XLNet_Ibra_epoch_3_combined_entire_weighted4_32_e5_linear.pkl")
    train_model(model, model_config, training_loader, validation_loader)
    eval_model(model, model_config, training_loader, validation_loader)
    generate_predictions(
         model,
         model_config,
         test_loader,
         "prediction.csv"
    )


if __name__ == "__main__":
    main()
