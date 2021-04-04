import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.ngram = 8
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        
        
    def tokenize(self, path):
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

        # # Tokenize file content
        # with open(path, 'r', encoding="utf8") as f:
        #     idss = []
        #     total = []
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         #ids = []
        #         for word in words:
        #             #ids.append(self.dictionary.word2idx[word])
        #             total.append(self.dictionary.word2idx[word])
        #         #idss.append(torch.tensor(ids).type(torch.int64))
        #     #ids = torch.cat(idss)
        
        # ngramDATA =[]
        # for i in range(len(total) - (self.ngram-1)):
        #     context = []
        #     for k in range(self.ngram-1):
        #         context.append(total[k+i])
                
        #     context_idxs = torch.tensor(context, dtype=torch.int64)
        #     target = torch.tensor([total[i+self.ngram-1]], dtype=torch.int64)
        #     ngramDATA.append((context_idxs,target))

        # return ngramDATA
