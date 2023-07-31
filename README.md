## Twitter  Datasets

Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip


The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

And place all these files in the twitter-datasets directory.


## Install python requirements

We used python 3.11.3 for this project. The required packages are listed in the requirements.txt file. To install them, run the following command:

``` 
pip install -r requirements.txt
```
### GloVe downloading

We used the standford GloVe for some of the baseline models. The corresponding pre-trained GloVe embeddings can be downloaded using this link:
https://nlp.stanford.edu/data/glove.6B.zip
The downloaded file should be put into the twitter-datasets directory.

### preprocessing of the dataset

Navigate to the preprocessing directory and run the preprocessing.py file.
This will generate multiple preprocessed csv files in the twitter-datasets directory.

```
python preprocessing.py
```

### Training the model

Navigate to the bert directory and run the main.py file.
This will first load and tokenize the data, then train the model and finally evaluate the model on the test data.
It will also generate a submission.csv file in the format required by Kaggle.

```
python main.py
```

### Ensemble Mehthod
We place all the prediction CSV files generated from different models into a folder named 'val_results/'. It will perform ensemble prediction and generate an ensemble_predictions.csv on the validation dataset.

```
python ensemble_prediction.py
```