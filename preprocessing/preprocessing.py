"""
This file contains the preprocessing steps for the tweets.
Run this file to generate the preprocessed files.

It will create the following files:

- no preprocessing just removing tabs
train_neg_full_notabs.csv
train_pos_full_notabs.csv

- removing emojis
train_neg_full_without_emoji.csv
train_pos_full_without_emoji.csv

- removing stopwords
train_neg_full_no_stopwords.csv
train_pos_full_no_stopwords.csv

- removing punctuation
train_neg_full_no_punctuation.csv
train_pos_full_no_punctuation.csv

- split hashtags
train_neg_full_split_hashtags.csv
train_pos_full_split_hashtags.csv

- spellcheck
train_neg_full_spellcheck.csv
train_pos_full_spellcheck.csv

- combined preprocessing with spellcheck, split hashtags and replacing emojis
train_neg_full_combined.csv
train_pos_full_combined.csv
"""

from tqdm import tqdm
import pandas as pd
import re

# from multiprocessing import Pool
from autocorrect import Speller
import pywordsegment
from concurrent.futures import ProcessPoolExecutor


def remove_tabs_dataset(filename):
    """
    Removes tabs from the file and replaces them with spaces.

    For the test data, the first comma is replaced with a tab.
    This way be can split the data into index and tweet by using the tab as a delimiter.
    , of ; can not be used as a delimiter because they are used in the tweets.
    """

    output_filename = filename.replace(".txt", "_notabs.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in tqdm(lines, desc=f"Preprocessing {filename}"):
        modified_line = line.replace("\t", " ").replace('"', "")

        if "test_data.txt" in filename:
            modified_line = modified_line.replace(",", "\t", 1)
        modified_lines.append(modified_line)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"Preprocessed file saved as {output_filename}")


EMOJI_DICT = {
    "<3": "Lovely",
    ":D": "Happy",
    ":P": "Playful",
    ":/": "Unsure",
    ":')": "Heartwarming",
    "=)": "Content",
    ";D": "Cheeky",
    ":|": "Neutral",
    ";P": "Teasing",
    ":-D": "Excited",
    ":\\": "Annoyed",
    ":-P": "Joking",
    ":]": "Happy",
    "=D": "Excited",
    ":'D": "Delighted",
    ":-/": "Puzzled",
    "=(": "Sad",
    ";/": "Disappointed",
    "0.0": "Surprised",
    "=P": "Amused",
    "=/": "Uneasy",
    "=]": "Optimistic",
    ":}": "Smug",
}


def replace_emojis(filename):
    """
    Replaces ascii emojis with adjectives that describe the emoji.
    """

    # method for replacing the emojis with adjectives
    def replace_emojis_in_tweet(dictionary, text):
        for key, value in dictionary.items():
            text = text.replace(key.lower(), value.lower())
        return text

    output_filename = filename.replace("_notabs.csv", "_without_emoji.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        line_without_emoji = replace_emojis_in_tweet(EMOJI_DICT, line)
        modified_lines.append(line_without_emoji)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file without emoji saved as {output_filename}")


def remove_stop_words(filename):
    """
    Romoves all stopwords from the tweets.
    """
    output_filename = filename.replace("_notabs.csv", "_no_stopwords.csv")

    SW = []
    with open("stopwords.txt", "r") as f:
        SW = f.read().splitlines()

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        if "test" in filename:
            idx, tweet = line.split("\t", 1)
            tokens = tweet.split(" ")
            line_without_sw = (
                f'{idx}\t{" ".join([token for token in tokens if token not in SW])}'
            )

        else:
            tokens = line.split(" ")
            line_without_sw = " ".join([token for token in tokens if token not in SW])

        modified_lines.append(line_without_sw)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file without stopwords saved as {output_filename}")


def remove_punctuation(text):
    text = re.sub(r"[^\w\s\'<>#]", " ", text)
    text = re.sub(r"\s+", " ", text) + "\n"
    return text


def remove_punctuation(filename):
    """
    Removes all punctuation from the tweets.
    """

    output_filename = filename.replace("_notabs.csv", "_no_punctuation.csv")

    def remove_punctuation_from_text(text):
        text = re.sub(r"[^\w\s\'<>#]", " ", text)
        text = re.sub(r"\s+", " ", text) + "\n"
        return text

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        if "test" in filename:
            idx, tweet = line.split("\t", 1)
            line_without_punctuation = f"{idx}\t{remove_punctuation_from_text(tweet)}"
        else:
            line_without_punctuation = remove_punctuation_from_text(line)

        modified_lines.append(line_without_punctuation)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file without punctuation saved as {output_filename}")


def split_hashtags(filename):
    """
    Removes all punctuation from the tweets.
    """

    output_filename = filename.replace("_notabs.csv", "_split_hashtags.csv")

    def split_hashtags_in_text(text):
        tokens = []
        for t in text.rstrip().split(" "):
            if t.startswith("#"):
                tokens.extend(pywordsegment.WordSegmenter.segment(t))
            else:
                tokens.append(t)
        return " ".join(tokens) + "\n"

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        if "test" in filename:
            idx, tweet = line.split("\t", 1)
            modified_lines.append(f"{idx}\t{split_hashtags_in_text(tweet)}")
        else:
            modified_lines.append(split_hashtags_in_text(line))

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file with split hashtags saved a {output_filename}")


def spellcheck_test_tweets(lines):
    spell = Speller(lang="en")

    modified_lines = []
    for line in tqdm(lines, miniters=200):
        index, line = line.split("\t", 1)
        tokens = line.rstrip().split(" ")
        modified_lines.append(f'{index}\t{" ".join((spell(t) for t in tokens))}\n')

    return modified_lines


def spellcheck_train_tweets(lines):
    spell = Speller(lang="en")

    modified_lines = []
    for line in tqdm(lines, miniters=200):
        tokens = line.rstrip().split(" ")
        modified_lines.append(" ".join((spell(t) for t in tokens)) + "\n")

    return modified_lines


def spellcheck(filename):
    """
    Runs spellcheck on the tweets.
    """

    output_filename = filename.replace("_notabs.csv", "_spellcheck.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    line_cnt = len(lines)
    chunk_size = line_cnt // 47
    chunks = [lines[i : i + chunk_size] for i in range(0, line_cnt, chunk_size)]
    with ProcessPoolExecutor() as p:
        if "test" in filename:
            print("test data")
            modified_lines = p.map(spellcheck_test_tweets, chunks)
        else:
            print("train data")
            modified_lines = p.map(spellcheck_train_tweets, chunks)

    output_lines = 0
    with open(output_filename, "w") as output_file:
        for chunk in modified_lines:
            output_lines += len(chunk)
            output_file.writelines(chunk)

    print(
        f"New file without combined preprocessing saved as {output_filename} (lines: {output_lines} / {line_cnt})"
    )
    assert output_lines == line_cnt


def combined_preprocess(lines):
    spell = Speller(lang="en")

    local_emoji_dict = {
        "<3": "Lovely",
        ":D": "Happy",
        ":P": "Playful",
        ":/": "Unsure",
        ":')": "Heartwarming",
        "=)": "Content",
        ";D": "Cheeky",
        ":|": "Neutral",
        ";P": "Teasing",
        ":-D": "Excited",
        ":\\": "Annoyed",
        ":-P": "Joking",
        ":]": "Happy",
        "=D": "Excited",
        ":'D": "Delighted",
        ":-/": "Puzzled",
        "=(": "Sad",
        ";/": "Disappointed",
        "0.0": "Surprised",
        "=P": "Amused",
        "=/": "Uneasy",
        "=]": "Optimistic",
        ":}": "Smug",
    }

    modified_lines = []
    for line in tqdm(lines, miniters=200):
        tokens = []
        for t in line.rstrip().split(" "):
            t = local_emoji_dict.get(t, t)

            if t.startswith("#"):
                tokens.extend(pywordsegment.WordSegmenter.segment(t))
            else:
                tokens.append(t)

        modified_lines.append(" ".join((spell(t) for t in tokens)) + "\n")

    print("done")
    return modified_lines


def combined_preprocess_test_data(lines):
    spell = Speller(lang="en")

    local_emoji_dict = {
        "<3": "Lovely",
        ":D": "Happy",
        ":P": "Playful",
        ":/": "Unsure",
        ":')": "Heartwarming",
        "=)": "Content",
        ";D": "Cheeky",
        ":|": "Neutral",
        ";P": "Teasing",
        ":-D": "Excited",
        ":\\": "Annoyed",
        ":-P": "Joking",
        ":]": "Happy",
        "=D": "Excited",
        ":'D": "Delighted",
        ":-/": "Puzzled",
        "=(": "Sad",
        ";/": "Disappointed",
        "0.0": "Surprised",
        "=P": "Amused",
        "=/": "Uneasy",
        "=]": "Optimistic",
        ":}": "Smug",
    }

    modified_lines = []
    for line in tqdm(lines, miniters=200):
        tokens = []
        index, line = line.split("\t", 1)
        for t in line.rstrip().split(" "):
            t = local_emoji_dict.get(t, t)

            if t.startswith("#"):
                tokens.extend(pywordsegment.WordSegmenter.segment(t))
            else:
                tokens.append(t)

        modified_lines.append(f'{index}\t{" ".join((spell(t) for t in tokens))}\n')

    print("done")
    return modified_lines


def gen_combined_preprocessed_file(filename):
    """
    combines the three successful preprocessing steps into one file.
    - split words
    - replace emojis
    - spellcheck
    """

    output_filename = filename.replace("_notabs.csv", "_combined.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    line_cnt = len(lines)
    chunk_size = line_cnt // 47
    chunks = [lines[i : i + chunk_size] for i in range(0, line_cnt, chunk_size)]
    with ProcessPoolExecutor() as p:
        if "test" in filename:
            print("test data")
            modified_lines = p.map(combined_preprocess_test_data, chunks)
        else:
            print("train data")
            modified_lines = p.map(combined_preprocess, chunks)

    output_lines = 0
    with open(output_filename, "w") as output_file:
        for chunk in modified_lines:
            output_lines += len(chunk)
            output_file.writelines(chunk)

    print(
        f"New file without combined preprocessing saved as {output_filename} (lines: {output_lines} / {line_cnt})"
    )
    assert output_lines == line_cnt


def replace_user_and_url(filename):
    """
    replaces <user> with @user and <url> with http
    """

    output_filename = filename.replace("_combined.csv", "_combined2.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    with open(output_filename, "w") as output_file:
        for line in lines:
            line = line.replace("<user>", "@user")
            line = line.replace("<url>", "http")

            output_file.write(line)


def main():
    ### remove tabs ###
    remove_tabs_dataset("../twitter-datasets/train_neg_full.txt")
    remove_tabs_dataset("../twitter-datasets/train_pos_full.txt")
    remove_tabs_dataset("../twitter-datasets/test_data.txt")

    ### replace emojis ###
    replace_emojis("../twitter-datasets/train_neg_full_notabs.csv")
    replace_emojis("../twitter-datasets/train_pos_full_notabs.csv")
    replace_emojis("../twitter-datasets/test_data_notabs.csv")

    ### remove stopwords ###
    remove_stop_words("../twitter-datasets/train_neg_full_notabs.csv")
    remove_stop_words("../twitter-datasets/train_pos_full_notabs.csv")
    remove_stop_words("../twitter-datasets/test_data_notabs.csv")

    ### remove punctuation ###
    remove_punctuation("../twitter-datasets/train_neg_full_notabs.csv")
    remove_punctuation("../twitter-datasets/train_pos_full_notabs.csv")
    remove_punctuation("../twitter-datasets/test_data_notabs.csv")

    ### split hashtags ###
    split_hashtags("../twitter-datasets/train_neg_full_notabs.csv")
    split_hashtags("../twitter-datasets/train_pos_full_notabs.csv")
    split_hashtags("../twitter-datasets/test_data_notabs.csv")

    ### spellcheck ###
    spellcheck("../twitter-datasets/train_neg_full_notabs.csv")
    spellcheck("../twitter-datasets/train_pos_full_notabs.csv")
    spellcheck("../twitter-datasets/test_data_notabs.csv")

    ### combined preprocessing ###
    # WARNING: this takes a long time to run
    gen_combined_preprocessed_file("../twitter-datasets/train_neg_full_notabs.csv")
    gen_combined_preprocessed_file("../twitter-datasets/train_pos_full_notabs.csv")
    gen_combined_preprocessed_file("../twitter-datasets/test_data_notabs.csv")

    ### replace user and url ###
    replace_user_and_url("../twitter-datasets/train_neg_full_combined.csv")
    replace_user_and_url("../twitter-datasets/train_pos_full_combined.csv")
    replace_user_and_url("../twitter-datasets/test_data_combined.csv")


if __name__ == "__main__":
    main()
