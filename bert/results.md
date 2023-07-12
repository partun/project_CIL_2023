## No preprocessing

- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Train Accuracy |
| ------ | -------------- | -------------- |
| 2      | 0.881          | 0.871          |
| 3      | 0.890          | 0.875          |

## Preprocessing
+ punctuation
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | -------------- |
| 2      |                | 0.876          |
| 3      | 0.893          | 0.881          |

+ spell check
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | -------------- |
| 2      |                | 0.865          |
| 3      | 0.883          | 0.871          |