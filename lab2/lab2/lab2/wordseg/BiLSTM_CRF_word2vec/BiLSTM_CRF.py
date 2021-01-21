from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.autograd as autograd

from wordseg.BiLSTM_CRF_word2vec.dataset import DataSet
from wordseg.BiLSTM_CRF_word2vec.vector_library import VectorLibrary


class BiLSTM_CRF(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, tag_to_ix, device=torch.device('cpu')):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.tag_to_ix = tag_to_ix

        # 增加两个虚拟标签：开始标签、结束标签
        self.START_TAG = '<START_TAG>'
        self.END_TAG = '<END_TAG>'
        self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)
        self.tag_to_ix[self.END_TAG] = len(self.tag_to_ix)
        self.START_TAG = self.tag_to_ix[self.START_TAG]
        self.END_TAG = self.tag_to_ix[self.END_TAG]

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
        embeds = torch.cat(sequence).view(size, 1, -1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(size, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _viterbi_decode(self, feats):
        back_pointers = []

        init_v_vars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_v_vars[0][self.START_TAG] = 0

        forward_vars = init_v_vars

        for feat in feats:
            bptrs = []
            viterbi_vars = []

            for next_tag in range(self.tagset_size):
                next_tag_vars = forward_vars + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_vars)
                bptrs.append(best_tag_id)
                viterbi_vars.append(next_tag_vars[0][best_tag_id].view(1))

            forward_vars = (torch.cat(viterbi_vars) + feat).view(1, -1)
            back_pointers.append(bptrs)

        # 以 <END_TAG> 为结尾
        terminal_var = forward_vars + self.transitions[self.END_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]

        for bptrs in reversed(back_pointers):
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


def argmax(vec):
    idx = torch.argmax(vec)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 全局初始化
torch.manual_seed(1)
epochs = 1

# 是否尝试开启GPU加速
enable_gpu = False


# 从标准输入读入字符串
def get_input_str(tip, default_value):
    x = str(input('%s (default=%s):' % (tip, default_value)))
    if (len(x.strip()) == 0):
        return default_value
    return x


# 从标准输入读入整数
def get_input_int(tip, default_value):
    x = input('%s (default=%s):' % (tip, default_value))
    if (not x.strip()):
        return default_value

    try:
        return int(x)
    except ValueError:
        return get_input_int(tip, default_value)


# 用测试集计算当前模型的误差
def accuracy(model, test_set):
    n = len(test_set)
    correct = 0
    total = 0
    for (i, (sentence, tags)) in enumerate(test_set):
        if (i % 1000 == 0):
            print('complete: %f%%' % (i / n * 100))
        sentence_in = test_set.prepare_word_sequence(sentence)
        tagsets = torch.tensor(test_set.prepare_tag_sequence(tags))
        _, outputs = model(sentence_in)
        outputs = torch.tensor(outputs)

        correct = correct + torch.sum(tagsets == outputs).item()
        total = total + len(sentence)

    return (correct / total)


# 输出模型对当前测试集的输出
def predict(model, vector_library, sentence):
    sentence = list(sentence)
    sentence_in = [vector_library[w] for w in sentence]
    _, tag_sets = model(sentence_in)
    ix_to_tag = dict((v, k) for k, v in model.tag_to_ix.items())
    tag_sets = [ix_to_tag[ix] for ix in tag_sets]
    result = ''
    for tag in tag_sets:
        result += tag
    return result


# 初始化模型，返回模型对象与训练集对象
def init_model(vector_library_path, training_set_path, enable_gpu):
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if (torch.cuda.is_available() and enable_gpu) else 'cpu')

    print("loading vector library...")
    vl = VectorLibrary(vector_library_path, device=device)

    print("loading training set...")
    training_set = DataSet(training_set_path, vl)

    # 创建 BiLSTM-CRF 网络，并在可行的情况下使用GPU加速
    model = BiLSTM_CRF(vl.vector_dim, 150, training_set.tag_to_ix, device)
    if (torch.cuda.is_available() and enable_gpu):
        model = model.cuda()

    return model, training_set


def train(model, training_set, epoch, enable_gpu):
    if (torch.cuda.is_available() and enable_gpu):
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if (torch.cuda.is_available() and enable_gpu) else 'cpu')
    print('working device = %s' % str(device))

    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
    n = len(training_set)

    print("start training...")
    for e in range(epoch):
        for (i, (sentence, tags)) in enumerate(training_set):
            if (i % 100 == 0):
                print('epoch = %d, progress = %f%%' % (e + 1, i / n * 100))

            model.zero_grad()

            sentence_in = training_set.prepare_word_sequence(sentence)
            tagsets = training_set.prepare_tag_sequence(tags)

            loss = model.loss(sentence_in, tagsets)

            loss.backward()
            optimizer.step()

        print('epoch = %d complete, loss = %f' % (e + 1, loss.item()))


if __name__ == '__main__':
    vector_library_path = '../../dataset/dataset1/vector_library.utf8'
    train_set_path = '../../dataset/dataset1/train.utf8'
    # train_set_path = '../dataset/dataset2/train.utf8'

    LSTM, train_set = init_model(vector_library_path, train_set_path, enable_gpu)
    train(LSTM, train_set, epochs, enable_gpu)
    LSTM.save('LSTM_1')
    # BiLSTM_CRF_word2vec = BiLSTM_CRF.load(save_path)
    # BiLSTM_CRF_word2vec.eval()
    # vector_library = VectorLibrary(vector_library_path)
    # train_set = DataSet(train_set_path, vector_library)
    # train(BiLSTM_CRF_word2vec, train_set, epochs, enable_gpu)
    # BiLSTM_CRF_word2vec.save(save_path)
