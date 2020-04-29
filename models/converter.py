import os


class TextConverter(object):
    def __init__(self, vocab_path, max_vocab=10000, min_freq=0):
        """Construct a text index converter.
        Args:
            text_path: txt file path.
            max_vocab: maximum number of words.
        """

        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab = [line.split('\t')[0] for line in f]
        self.vocab = vocab
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return self.word_to_int_table['<unk>']

    def int_to_word(self, index):
        if index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return arr

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)