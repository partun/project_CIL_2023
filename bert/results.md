## No preprocessing

- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      | 0.881          | 0.871               |
| 3      | 0.890          | 0.875               |

## Punctuation
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      |                | 0.876               |
| 3      | 0.893          | 0.881               |

## spell check
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      |                | 0.865               |
| 3      | 0.883          | 0.871               |

## hashtag
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      | 0.881          | 0.869               |
| 3      | 0.887          | 0.873               |


- batch size = 16
- learning_rate = 0.00001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      | 0.887          | 0.878               |
| 3      | 0.895          | 0.883               |

## no emojis

- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = noemoji
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.871               |
| 2      | 0.887          | 0.879               |


## no stopwords

- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = nostopwords
- force_reload_dataset = False


| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.857               |
| 2      | 0.874          | 0.865               |

## no punctuation
- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = nopunctuation
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.858               |
| 2      | 0.875          | 0.867               |


## no stopwords & hashtag split
- batch size = 16
- learning_rate = 0.0001
- tokenizer_model = "distilbert-base-uncased"
- nn_model = "prajjwal1/bert-mini"
- binary cross entropy loss function

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      | 0.877          | 0.864               |
| 3      | 0.881          | 0.868               |


## full
- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = full
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.871               |
| 2      | 0.887          | 0.878               |
