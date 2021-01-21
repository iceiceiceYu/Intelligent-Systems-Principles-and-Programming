import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from wordseg.BiLSTM_CRF.dataset import DataSet


torch.manual_seed(1)


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, word_to_ix=None, tag_to_ix=None, device=torch.device('cpu')):
        super(BiLSTM_CRF, self).__init__()
        if tag_to_ix is None:
            tag_to_ix = {'B': 0, 'E': 1, 'I': 2, 'S': 3}
        self.word_to_ix = word_to_ix
        if word_to_ix is None:
            self.vocab_size = 4179
        else:
            self.vocab_size = len(word_to_ix)
        self.hidden_dim = hidden_dim
        self.device = device
        self.tag_to_ix = tag_to_ix
        self.word_embeds = nn.Embedding(self.vocab_size, input_dim)

        # 增加两个虚拟标签：开始标签、结束标签
        self.END_TAG = '<END_TAG>'
        self.START_TAG = '<START_TAG>'
        self.tag_to_ix[self.END_TAG] = len(self.tag_to_ix)
        self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)

        self.END_TAG = self.tag_to_ix[self.END_TAG]
        self.START_TAG = self.tag_to_ix[self.START_TAG]

        self.tagset_size = len(self.tag_to_ix)

        # 初始化网络参数
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.END_TAG] = -10000
        self.hidden = self.init_hidden()

        self.to(self.device)


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
                torch.randn(2, 1, self.hidden_dim // 2, device=self.device))


    def _get_lstm_features(self, sequence):
        size = len(sequence)
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sequence).view(size, 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(size, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.START_TAG] = 0

        forward_vars = init_vvars

        for feat in feats:
            bptrs = []
            viterbivars = []

            for next_tag in range(self.tagset_size):
                next_tag_vars = forward_vars + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_vars)
                bptrs.append(best_tag_id)
                viterbivars.append(next_tag_vars[0][best_tag_id].view(1))

            forward_vars = (torch.cat(viterbivars) + feat).view(1, -1)
            backpointers.append(bptrs)

        # 以 <END_TAG> 为结尾
        terminal_var = forward_vars + self.transitions[self.END_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]

        for bptrs in reversed(backpointers):
            best_tag_id = bptrs[best_tag_id]
            best_path.append(best_tag_id)

        # 以 <START_TAG> 为开头
        start = best_path.pop()
        assert start == self.START_TAG

        best_path.reverse()

        return path_score, best_path


    def forward(self, sequence):
        lstm_feats = self._get_lstm_features(sequence)
        score, best_seq = self._viterbi_decode(lstm_feats)
        return score, best_seq


    def loss(self, sequence, tags):
        lstm_feats = self._get_lstm_features(sequence)
        forward_score = self._forward_alg(lstm_feats)
        gold_score = self._score_sentence(lstm_feats, tags)
        return forward_score - gold_score


    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=self.device)

        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long, device=self.device), tags])

        for (i, feat) in enumerate(feats):
            emit_score = feat[tags[i + 1]]
            trans_score = self.transitions[tags[i + 1], tags[i]]
            score = score + emit_score + trans_score

        score = score + self.transitions[self.END_TAG, tags[-1]]
        return score


    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_alphas[0][self.START_TAG] = 0.

        forward_vars = init_alphas
        for feat in feats:
            alphas = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_vars + trans_score + emit_score
                alphas.append(log_sum_exp(next_tag_var).view(1))

            forward_vars = torch.cat(alphas).view(1, -1)

        terminal_var = forward_vars + self.transitions[self.END_TAG]
        return log_sum_exp(terminal_var)


    def save(self, path):
        torch.save(self.state_dict(), f'../../model/LSTM/{path}.model')


    def load(self, path):
        self.load_state_dict(torch.load(f'model/LSTM/{path}.model'))


# 输出模型对当前测试集的输出
def predict(model, sentence):
    sentence = list(sentence)
    for w in sentence:
        if w in model.word_to_ix:
            pass
        else:
            model.word_to_ix[w] = np.random.randint(0, 10, size=1)
    sentence_in = [model.word_to_ix[w] for w in sentence]
    sentence_in = torch.tensor(sentence_in, dtype=torch.long)
    _, tagsets = model(sentence_in)
    ix_to_tag = dict((v, k) for k, v in model.tag_to_ix.items())
    tagsets = [ix_to_tag[ix] for ix in tagsets]
    result = ''
    for tag in tagsets:
        result += tag
    return result


def argmax(vec):
    idx = torch.argmax(vec)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


if __name__ == '__main__':
    train = True
    epoch = 10
    training_set = DataSet('../../dataset/dataset2/train.utf8')
    # vl = VectorLibrary('vector_library.utf8')
    if train:
        # model = BiLSTM_CRF(vl.vector_dim, 150, training_set.tag_to_ix)
        model = BiLSTM_CRF(300, 150, training_set.word_to_ix, training_set.tag_to_ix, )
        # model.load()
        model.eval()
        optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        for i in range(epoch):
            for (j, (sentence, tags)) in enumerate(training_set):
                if j % 100 == 0:
                    print('epoch = %d, progress = %f%%' % (i + 1, j / len(training_set) * 100))
                model.zero_grad()
                if len(sentence) != 0:
                    sentence_in = training_set.prepare_sequence(sentence)
                    tagsets = training_set.prepare_tag_sequence(tags)
                    loss = model.loss(sentence_in, tagsets)
                    loss.backward()
                    optimizer.step()
            print('epoch = %d complete, loss = %f' % (i + 1, loss.item()))
            model.save(f'LSTM_2_epoch{i + 1}')
