import torch


class DataSet:
    def __init__(self, path, encoding='utf8'):
        self.tag_to_ix = {}
        self.word_to_ix = {}
        self.data = []

        with open(path, 'r', encoding=encoding) as f:
            rows = f.readlines()
            sentence = ([], [])
            for row in rows:
                i = row.strip()
                if len(i) > 0:
                    tag = 'NIL'
                    i = i.split()
                    if len(i) == 1:
                        sentence[0].append(i[0])
                        sentence[1].append(tag)
                    else:
                        word, tag = i
                        sentence[0].append(word)
                        sentence[1].append(tag)
                    if word not in self.word_to_ix:
                        self.word_to_ix[word] = len(self.word_to_ix)
                    if tag not in self.tag_to_ix:
                        self.tag_to_ix[tag] = len(self.tag_to_ix)

                else:
                    self.data.append(sentence)
                    sentence = ([], [])

            if len(sentence) > 0:
                self.data.append(sentence)


    def prepare_sequence(self, seq):
        idxs = [self.word_to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)


    def prepare_tag_sequence(self, seq):
        idxs = [self.tag_to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)


    def tag_idxs_to_sequence(self, idxs):
        self.ix_to_tag = dict((v, k) for k, v in self.tag_to_ix.items())
        return [self.ix_to_tag[ix] for ix in idxs]


    def __getitem__(self, key):
        return self.data[key]


    def __len__(self):
        return len(self.data)
