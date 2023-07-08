import pywordsegment


# load customized stopwords based on nltk stopwords
SW = []
with open("stopwords.txt", 'r') as f:
    SW = f.read().splitlines() 
INPUT_FILE = '../twitter-datasets/train_pos.txt'           # files to rm hashtag & sw
OUTPUT_FILE = '../twitter-datasets/train_pos_swht.txt'     # output file


def process_hashtag_stop_words(text):
    """
    input: a tweet string
    output: a tweet string w/ hashtag handled and stopwords removed
    """
    clean_list = []
    raw_list = text.rstrip().split(" ")
    for word in raw_list:
        temp_list = []
        if '#' in word:
            temp_list = pywordsegment.WordSegmenter.segment(word) # split hashtag
            clean_list.extend(temp_list)
        else:
            clean_list.append(word)
    
    line = ' '.join(word for word in clean_list if word not in SW) # rm stopwords
    return line


def clean_tweets(intput, output):
    modified_tweets = []

    with open(intput, "r", encoding="utf-8") as f:
        for line in f:
            line_wo_swht = process_hashtag_stop_words(line)
            modified_tweets.append(line_wo_swht + '\n')

    with open(output, 'w') as f:
        f.writelines(modified_tweets)


# Example use
# clean_tweets(INPUT_FILE, OUTPUT_FILE)
