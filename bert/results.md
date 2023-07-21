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


## combined: no emoji + hashtag splitting + spell check

- tokenizer_model  = distilbert-base-uncased
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = False


| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.870               |
| 2      | 0.886          | 0.878               |
| 3      | -              | 0.881               |
| 3      | 0.899          | 0.884               |


## combined 3: no emoji + hashtag splitting + spell check

- tokenizer_model  = prajjwal1/bert-mini
- max_length       = 40
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = True

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.869               |
| 2      | 0.883          | 0.875               |


## combined 4: no emoji + hashtag splitting + spell check

- tokenizer_model  = distilbert-base-uncased
- max_length       = 45
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 64
- valid_batch_size = 64
- epochs           = 2
- learning_rate    = 0.0001
- dataset_type     = combined
- force_reload_dataset = True


| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.871               |
| 2      | 0.895          | 0.879               |
| 3      | -              | 0.881               |
| 4      | 0.914          | 0.883               |


## combined 5: no emoji + hashtag splitting + spell check

- tokenizer_model  = distilbert-base-uncased
- max_length       = 45
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 64
- valid_batch_size = 64
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = False


| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.861               |
| 2      | -              | 0.870               |


## combined 6: no emoji + hashtag splitting + spell check

- tokenizer_model  = distilbert-base-uncased
- max_length       = 45
- nn_model         = prajjwal1/bert-small
- device           = cuda
- train_batch_size = 64
- valid_batch_size = 64
- epochs           = 2
- learning_rate    = 0.0001
- dataset_type     = combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.877               |
| 2      | 0.901          | 0.880               |
| 3      | -              | 0.883               |
| 4      | 0.922          | 0.883               |


## combined 7:

- tokenizer_model  = distilbert-base-uncased
- max_length       = 45
- nn_model         = prajjwal1/bert-tiny
- device           = cuda
- train_batch_size = 64
- valid_batch_size = 64
- epochs           = 2
- learning_rate    = 0.0001
- dataset_type     = combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.858               |
| 2      | 0.877          | 0.864               |
| 3      | -              | 0.868               |
| 4      | 0.893          | 0.870               |


## combined 8:

- tokenizer_model  = distilbert-base-uncased
- max_length       = 45
- dropout          = 0.5
- nn_model         = prajjwal1/bert-mini
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- epochs           = 2
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | -              | 0.866               |
| 2      | 0.881          | 0.875               |
| 3      | -              | 0.878               |
| 4      | 0.894          | 0.882               |
| 5      | 0.886          | 0.883               |
| 6      | 0.890          | 0.885               |
| 7      | 0.894          | 0.886               |
| 8      | 0.897          | 0.884               |
| 9      | 0.900          | 0.886               |
| 10     | 0.903          | 0.886               |

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


### RoBERTa run 2
- tokenizer_model  = roberta-base
- max_length       = 45
- nn_model         = roberta-base
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- epochs           = 4
- learning_rate    = 1e-05
- dataset_type     = combined
- force_reload_dataset = True

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.884          | 0.897               |
| 2      | 0.904          | 0.903               |
| 3      | 0.914          | 0.903               |
| 4      | 0.924          | 0.904               |


## XLNet

### 1. Fine-tune the entire model (full) + linear(7.15)
#### - representation: last hidden layer
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

### 2. Fine-tune the entire model + linear 1 (7.18)
#### - representation: last hidden layer
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 16
- valid_batch_size = 16
- learning_rate    = 1e-05
- dataset_type     = Combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.905          | 0.893               |
| 2      | 0.922          | 0.898               |
| 3      | 0.934          | 0.900               |

### 3. Fine-tune the entire model + linear 2 (7.18)
#### - representation: last hidden layer
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- dataset_type     = Combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.900          | 0.894               |
| 2      | 0.922          | 0.899               |

### 4. Fine-tune the entire model + linear (7.21)
#### - representation: weighted average last 4
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- dataset_type     = Combined
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.905          | 0.894               |
| 2      | 0.921          | 0.899               |
| 3      | 0.934          | 0.901               |

### 5. Fine-tune the entire model + cnn (7.21)
#### - representation: last 4 layers as 4 channels to cnn
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- dataset_type     = Combined
- filter number    = 256
- kernel_size      = 3
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.906          | 0.895               |
| 2      | 0.922          | 0.899               |

### 6. Fine-tune the entire model + cnn (7.21)
#### - representation: last 4 to cnn + middle 4 to cnn, then concat
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- dataset_type     = Combined
- filter number = 256
- kernel_size = 3
- force_reload_dataset = False

| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.903          | 0.892               |
| 2      | 0.921          | 0.898               |

### 7. Fine-tune the last 4 layers (7.15)
#### - representation: last 4 hidden layer
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

### 8. Fine-tune the last 4 layers (7.15)
#### - representation: weighted average last 4 hidden layers
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


### 9. Fine-tune the last 6 layers (7.18)
#### - representation: concatenate last 4 layers (Combined)
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- dataset_type     = Combined
- force_reload_dataset = False
  
| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.811          | 0.810               |
| 2      | 0.814          | 0.814               |

### 10. Fine-tune the last 6 layers (7.21)
#### - representation: last 4 layers as 4 channels to cnn
- tokenizer_model  = xlnet-base-cased
- nn_model         = Ibrahim-Alam/finetuning-xlnet-base-cased-on-tweet_sentiment_binary
- device           = cuda
- train_batch_size = 32
- valid_batch_size = 32
- learning_rate    = 1e-05
- filter_number    = 256
- kernel_size      = 3
- dataset_type     = Combined
- force_reload_dataset = False
  
| Epochs | Train Accuracy | Validation Accuracy |
| ------ | -------------- | ------------------- |
| 1      | 0.883          | 0.879               |
| 2      | 0.894          | 0.885               |
| 3      | 0.902          | 0.889               |
| 4      | 0.909          | 0.890               |




