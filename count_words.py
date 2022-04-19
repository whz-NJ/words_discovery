#!/root/anaconda3/bin/python
import os
import re

def get_files(dir, file_name_pattern):
    all_files = []
    file_name_pattern = re.compile(file_name_pattern)
    for root, dirs, files in os.walk(dir):
        for path in files:
            file_path = os.path.join(root, path)
            if file_name_pattern.match(file_path):
                all_files.append(file_path)
    return all_files

new_words_map = {}
words_map = {}
files = get_files(".", ".+words_seq.txt")
for file_path in files:
    new_words = True
    print("begin to count words in " + file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l.startswith('#'):
                continue
            if l:
                fields = l.split('\t')
                if len(fields) == 2:
                    word = fields[0]
                    count = fields[1]
                    if count == '词频':
                        if word == '新词':
                            new_words = True
                        else:
                            new_words = False
                        continue
                    if new_words:
                        if word in new_words_map:
                            new_words_map[word] += int(count)
                        else:
                            new_words_map[word] = int(count)
                    else:
                        if word in words_map:
                            words_map[word] += int(count)
                        else:
                            words_map[word] = int(count)

with open('words_count.txt', 'w', encoding='utf-8') as f:
    f.write("新词\t词频\n")
    # new_words_map.items() 每个元素变成2个item的元祖
    for word, freq_sort in sorted(new_words_map.items(), key=lambda x: x[1], reverse=True):
        f.write(word + '\t' + str(freq_sort) + '\n')

    f.write("\n已知词\t词频\n")
    for word, freq in sorted(words_map.items(), key=lambda x: x[1], reverse=True):
        f.write(word + '\t' + str(freq) + '\n')

