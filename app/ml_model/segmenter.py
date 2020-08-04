import gensim
import nltk
import numpy as np
import torch


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


class Segmenter:
    def __init__(self, model_path, word2vec_path, seg_threshold):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seg_threshold = seg_threshold
        self._initialize(model_path, word2vec_path)

    def _initialize(self, model_path: str, word2vec_path: str):
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        try:
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt')
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def _tokenize(self, word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['UNK']

    def segment(self, paragraph: str, threshold=None):
        sentences = self.sent_tokenizer.tokenize(paragraph)
        tensored = []
        for sentence in sentences:
            words = self.word_tokenizer.tokenize(sentence)
            tensored.append(torch.FloatTensor([self._tokenize(word) for word in words]))
        output = self.model([tensored])
        output_prob = softmax(output.data.cpu().numpy())
        if threshold:
            output_seg = output_prob[:, 1] > threshold
        else:
            output_seg = output_prob[:, 1] > self.seg_threshold
        segments = [""]
        for i in range(len(sentences) - 1):
            segments[-1] += sentences[i] + " "
            if output_seg[i] == 1:
                segments.append("")
        segments[-1] += sentences[-1]
        return segments
