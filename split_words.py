#!/root/anaconda3/bin/python
import re
import numpy as np
from collections import ChainMap
from collections import defaultdict
from pyhanlp import *

class TrieNode():
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.char = ''
        self.count = 0

class Trie():
    '''
    实现如下功能：
    1. 记录总的词频数：total_count
    2. 输入单词，返回其词频：get_freq
    3. 输入单词，返回其子节点的所有char和相应count：get_children_chars
    4. 迭代器返回插入trie的所有单词及其count：get_all_words
    '''

    def __init__(self):
        self.root = TrieNode()
        self.total_count = 0

    def insert(self, text):
        node = self.root
        for c in text:
            node = node.children[c]
            node.char = c
        node.count += 1
        self.total_count += 1

    def get_all_words(self):
        q = [('', self.root)]
        while q:
            prefix, node = q.pop(0)
            for child in node.children.values():
                if child.count:
                    yield prefix + child.char, child.count
                q.append((prefix + child.char, child))

    def get_freq(self, text):
        node = self.root
        for c in text:
            if c not in node.children:
                return 0
            node = node.children[c]
        return node.count

    def get_children_chars(self, text):
        node = self.root
        for c in text:
            if c not in node.children:
                return []
            node = node.children[c]
        return [(k.char, k.count) for k in node.children.values()]

STR_PUNCTUATIONS = ".:;?! \t\r\n~,-_()[\]<>。：；？！~，、——（）【】《》＃＊＝＋/｜‘’“”￥#*=+|'\"^$%`"
STR_TONES = '你我它他啊哦呃吗呀吧噢嗯哎唉哪呵嘿嗨哼'
RE_DATE = r'[0-9一二三四五六七八九十]{1,4}[年月日]([0-9一二三四五六七八九十]{1,4}[年月日])?'
RE_DIGITS = r'^[0-9一二三四五六七八九十]+$'

class NewWords():
    def __init__(self):
        self.trie = Trie()
        self.trie_reversed = Trie()
        self.word_info = defaultdict(dict)
        self.NGRAM = 6
        self.WORD_MIN_LEN = 3
        self.WORD_MIN_FREQ = 2
        self.WORD_MIN_PMI = 8
        self.WORD_MIN_NEIGHBOR_ENTROPY = 1
        self.tones = set(STR_TONES)
        self.punctuations = set(STR_PUNCTUATIONS)
        self.re_date = re.compile(RE_DATE)
        self.re_digits = re.compile(RE_DIGITS)

    def skip(self, word):
        for ch in word:
            if ch in self.tones:
                return True
        matched = self.re_date.match(word)
        if matched:
            return True
        matched = self.re_digits.match(word)
        if matched:
            return True

    def parse(self, text):
        words = self.n_gram_words(text)
        self.build_trie(words)
        self.get_words_pmi()
        self.get_words_entropy()

    def n_gram_words(self, text):
        text = text.lower() #忽略大小写
        words = []
        segment = []
        for c in text:
            if c not in self.punctuations:
                segment.append(c)
                continue
            if len(segment) == 0:
                continue
            segment = ''.join(segment)
            ngram = min(self.NGRAM, len(segment))
            for i in range(1, ngram + 1):
                words += [segment[j:j + i] for j in range(len(segment) - i + 1)]
            segment = []
        return words

    def build_trie(self, words):
        for word in words:
            self.trie.insert(word)
            self.trie_reversed.insert(word[::-1])

    def get_words_pmi(self):
        for word, count in self.trie.get_all_words():
            if len(word) < self.WORD_MIN_LEN or count < self.WORD_MIN_FREQ:
                continue
            pmi = min(
                [count * self.trie.total_count / self.trie.get_freq(word[:i]) / self.trie.get_freq(word[i:]) for i in
                 range(1, len(word)) \
                 if self.trie.get_freq(word[:i]) and self.trie.get_freq(word[i:])])
            pmi = np.log2(pmi)
            self.word_info[word]['pmi'] = pmi
            self.word_info[word]['freq'] = count

    def calculate_entropy(self, char_list):
        if not char_list:
            return 0
        num = sum([v for k, v in char_list])
        entropy = (-1) * sum([(v / num) * np.log2(v / num) for k, v in char_list])
        return entropy

    def get_words_entropy(self):
        for k, v in self.word_info.items():
            right_char = self.trie.get_children_chars(k)
            right_entropy = self.calculate_entropy(right_char)

            left_char = self.trie_reversed.get_children_chars(k[::-1])
            left_entropy = self.calculate_entropy(left_char)

            entropy = min(right_entropy, left_entropy)
            self.word_info[k]['entropy'] = entropy

    def candidates(self, sortby='pmi'):
        res = [(k, v) for k, v in self.word_info.items() if
               v['pmi'] >= self.WORD_MIN_PMI and v['entropy'] >= self.WORD_MIN_NEIGHBOR_ENTROPY]
        res = sorted(res, key=lambda x: x[1][sortby], reverse=True)
        for k, v in res:
            if self.skip(k):
                continue
            yield k, v

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("需要输入文件路径、提取词数")
        exit(1)

    word_freq_map = {}
    new_word_freq_map = {}
    discover = NewWords()
    file_path = sys.argv[1]
    old_words_num = int(sys.argv[2])

    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l.startswith('#'):
                continue
            if l:
                sentences.append(l + '\n')

    # words = jieba.lcut(sentence)
    # for word in words:
    #     if(len(word) > 1 and words != '\r\n' and words != '\n'):
    #         if word in word_freq_map:
    #             word_freq_map[word] = word_freq_map[word] + 1
    #         else:
    #             word_freq_map[word] = 1

    for sentence in sentences:
        lines = HanLP.extractPhrase(sentence, old_words_num)
        for word in lines:
            if (len(word) > 1 and lines != '\r\n' and lines != '\n'):
                if word in word_freq_map:
                    word_freq_map[word] = word_freq_map[word] + 1
                else:
                    word_freq_map[word] = 1

    sentences = ''.join(sentences)
    discover.parse(sentences)
    sorted_by = 'pmi'
    for k, v in discover.candidates(sorted_by):
        if k.strip() not in word_freq_map:
            new_word_freq_map[k] = [v['freq'],v[sorted_by]]

    # words_freq = sorted(dict(ChainMap(word_freq_map, new_word_freq_map)).items(), key=lambda x: x[1], reverse=True)
    if file_path.rfind('.') > 0:
        out_file_path = file_path[0:file_path.rfind('.')] + '_words_seq.txt'
    else:
        out_file_path = file_path + '_words_seq.txt'
    with open(out_file_path, 'w', encoding='utf-8') as f:
        f.write("新词\t词频\n")
        for word, freq_sort in sorted(new_word_freq_map.items(), key=lambda x: x[1][0]*100+x[1][1], reverse=True):
            f.write(word + '\t' + str(freq_sort[0]) + '\n')

        f.write("\n已知词\t词频\n")
        for word, freq in sorted(word_freq_map.items(), key=lambda x: x[1], reverse=True):
            f.write(word + '\t' + str(freq) + '\n')



