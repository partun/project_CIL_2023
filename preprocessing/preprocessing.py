from tqdm import tqdm
import pandas as pd
import re


def preprocess_file(filename):
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


emoji_dict = {
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


# method for replacing the emojis with adjectives
def replace_emoji(dictionary, text):
    for key, value in dictionary.items():
        text = text.replace(key.lower(), value.lower())
    return text


def get_file_without_emoji(filename):
    output_filename = filename.replace("_notabs.csv", "_without_emoji.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        line_without_emoji = replace_emoji(emoji_dict, line)
        modified_lines.append(line_without_emoji)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file without emoji saved as {output_filename}")


def remove_stop_words(filename):
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
    #     print(text)
    text = re.sub(r"[^\w\s\'<>#]", " ", text)
    text = re.sub(r"\s+", " ", text) + "\n"
    #     print(text)
    return text


def gen_files_without_punctuation(filename):
    output_filename = filename.replace("_notabs.csv", "_no_punctuation.csv")

    with open(filename, "r") as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        if "test" in filename:
            idx, tweet = line.split("\t", 1)
            line_without_punctuation = f"{idx}\t{remove_punctuation(tweet)}"
        else:
            line_without_punctuation = remove_punctuation(line)

        modified_lines.append(line_without_punctuation)

    with open(output_filename, "w") as output_file:
        output_file.writelines(modified_lines)

    print(f"New file without punctuation saved as {output_filename}")


if __name__ == "__main__":
    # preprocess_file("../twitter-datasets/train_neg.txt")
    # preprocess_file("../twitter-datasets/train_pos.txt")
    # preprocess_file("../twitter-datasets/train_pos_full.txt")
    # preprocess_file("../twitter-datasets/train_neg_full.txt")
    # preprocess_file("../twitter-datasets/test_data.txt")

    files = [
        "../twitter-datasets/train_neg_full_notabs.csv",
        "../twitter-datasets/train_pos_full_notabs.csv",
        "../twitter-datasets/test_data_notabs.csv",
    ]  # Update with your filenames

    for filename in files:
        # get_file_without_emoji(filename)
        # remove_stop_words(filename)
        gen_files_without_punctuation(filename)
