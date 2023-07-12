from autocorrect import Speller
import re
import concurrent.futures

spell = Speller(lang="en")
WORD = re.compile(r"[\w'<>,.!#]+")


def reTokenize(doc):
    tokens = WORD.findall(doc)
    return tokens


def spell_check(text):
    return ' '.join([spell(w) for w in reTokenize(text)])


def process_lines(lines):
    modified_lines = []
    for line in lines:
        line_spell_check = spell_check(line) + '\n'
        modified_lines.append(line_spell_check)
        if len(modified_lines) % 1000 == 0:
            print(len(modified_lines))
    return modified_lines


def get_file_without_spell_check(filename):
    output_filename = filename.replace(".txt", "_with_spell_check.txt")
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()

    num_lines = len(lines)
    chunk_size = 100000
    chunks = [lines[i:i + chunk_size] for i in range(0, num_lines, chunk_size)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_lines, chunks)

    modified_lines = []
    processed_chunks = 0
    for result in results:
        modified_lines.extend(result)
        processed_chunks += 1
        print(f"Processed {processed_chunks}/{len(chunks)} chunks")

    with open(output_filename, 'w') as output_file:
        output_file.writelines(modified_lines)

    print(f"New file with spell check saved as {output_filename}")


files = [
    "../twitter-datasets/train_neg_full.txt",
    "../twitter-datasets/train_pos_full.txt"
]  # Update with your filenames

for filename in files:
    get_file_without_spell_check(filename)
