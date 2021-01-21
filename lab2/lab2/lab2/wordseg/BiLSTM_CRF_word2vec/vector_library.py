import torch


# 向量库
class VectorLibrary:
    def __init__(self, filename, encoding='utf8', device=torch.device('cpu')):
        self.data = dict()
        self.device = device
        with open(filename, 'r', encoding=encoding) as f:
            rows = f.readlines()
            self.vocab_size, self.vector_dim = tuple(map(int, rows[0].split()))
            for row in rows[1:]:
                arr = row.split()
                word = arr[0]
                vector = torch.tensor(list(map(float, arr[1:])), device=device)
                self.data[word] = vector


    def __getitem__(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:
            self.data[key] = torch.randn(self.vector_dim, device=self.device)
            return self.data[key]


    def __len__(self):
        return len(self.data)