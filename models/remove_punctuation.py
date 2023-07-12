import re
def remove_punctuation(text):
#     print(text)
    text = re.sub(r'[^\w\s\'<>#]', ' ', text)
    text = re.sub(r'\s+', ' ', text) +'\n'
#     print(text)
    return text

def get_file_without_punctuation(filename):
    output_filename = filename.replace(".txt", "_without_punctuation.txt")        
    
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()
    
    modified_lines = []
    for line in lines:
        line_without_emoji = remove_punctuation(line)
        modified_lines.append(line_without_emoji)
    
    with open(output_filename, 'w') as output_file:
        output_file.writelines(modified_lines)
    
    print(f"New file without emoji and punctuation saved as {output_filename}")

files = ["../twitter-datasets/train_neg_full.txt", "../twitter-datasets/train_pos_full.txt"]  # Update with your filenames

for filename in files:
    get_file_without_punctuation(filename)