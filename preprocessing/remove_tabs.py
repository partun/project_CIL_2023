from tqdm import tqdm


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


if __name__ == "__main__":
    preprocess_file("../twitter-datasets/train_neg.txt")
    preprocess_file("../twitter-datasets/train_pos.txt")
    preprocess_file("../twitter-datasets/train_pos_full.txt")
    preprocess_file("../twitter-datasets/train_neg_full.txt")
    preprocess_file("../twitter-datasets/test_data.txt")
