import torch
from transformers import DebertaModel, DebertaTokenizer

class DeBERTaClass(torch.nn.Module):
    def __init__(self, model_name="deberta-v3-base"):
        super(DeBERTaClass, self).__init__()
        self.l1 = DebertaModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        # self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
        self.l4 = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        inputs = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": token_type_ids
        }
        output = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]
        # output = self.l2(output)
        output = self.l3(output[:, 0, :])
        output = self.l4(output)
        return output
