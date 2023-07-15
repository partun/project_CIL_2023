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

## No punctuation
- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 3
- learning_rate    = 1e-05
- dataset_type     = without_punctuation
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      |                | 0.865               |
| 3      | 0.883          | 0.871               |

## spell check
- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 3
- learning_rate    = 1e-05
- dataset_type     = with_spell_check
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 2      |                | 0.876               |
| 3      | 0.893          | 0.881               |

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

## RoBERTa
- tokenizer_model  = roberta-base
- nn_model         = roberta-base
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 1
- learning_rate    = 1e-05
- dataset_type     = full
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.911          | 0.899               |

## XLNet

### 1. Fine-tune the entire model
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- learning_rate    = 1e-05
- dataset_type     = full
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.909          | 0.896               |

### 2. Only fine-tune the last 4 hidden layers (totally 12 layers)
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.877          | 0.873               |
| 2      | 0.886          | 0.880               |
| 3      | 0.893          | 0.883               |

### 3. Only fine-tune the last 4 hidden layers, weighted average them as the final representation
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 1
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.874          | 0.871               |
| 2      |                | 0.878               |
| 3      | 0.892          | 0.883               |
