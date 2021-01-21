import numpy as np


tag2number = {'S': 0, 'B': 1, 'I': 2, 'E': 3}
number2tag = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}
epochs = 20


class CRF:
    def __init__(self):
        self.score_map = {'Test': 1}
        self.unigram = None
        self.bigram = None
        self.init_template()


    def init_template(self):
        self.unigram = [
            [-2],
            [-1],
            [0],
            [1],
            [2],
            [-2, -1],
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 2]
        ]
        self.bigram = [
            [-2],
            [-1],
            [0],
            [1],
            [2],
            [-2, -1],
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 2]
        ]


    def predict(self, sentence):
        length = len(sentence)
        status_from = np.zeros((4, length), dtype=str)
        max_score = np.zeros((4, length))
        # find = {0: 'B', 1: 'I', 2: 'E', 3: 'S'} # number2tag
        # find_idx = {'B': 0, 'I': 1, 'E': 2, 'S': 3} # tag2number
        for i in range(length):
            for j in range(4):
                status = number2tag[j]
                if i == 0:
                    unigram_score = self.get_unigram_score(sentence, 0, status)
                    bigram_score = self.get_bigram_score(sentence, 0, ' ', status)
                    max_score[j][0] = unigram_score + bigram_score
                    status_from[j][0] = ''
                else:
                    scores = np.zeros((1, 4))
                    for k in range(4):
                        pre_status = number2tag[k]
                        trans_score = max_score[k][i - 1]
                        unigram_score = self.get_unigram_score(sentence, i, status)
                        bigram_score = self.get_bigram_score(sentence, i, pre_status, status)
                        scores[0][k] = trans_score + unigram_score + bigram_score
                    max_score[j][i] = np.max(scores[0])
                    status_from[j][i] = number2tag[np.argmax(scores[0])]
        res_buf = [0] * length
        score_buf = np.zeros((1, 4))
        for i in range(4):
            score_buf[0][i] = max_score[i][length - 1]
        res_buf[length - 1] = number2tag[np.argmax(score_buf[0])]
        for back_idx in range(length - 2, -1, -1):
            res_buf[back_idx] = status_from[tag2number[res_buf[back_idx + 1]]][back_idx + 1]
        temp = ''
        for i in range(length):
            temp = temp + res_buf[i]
        return temp


    def get_unigram_score(self, sentence, pos, status):
        unigram_score = 0
        for i in range(len(self.unigram)):
            key = get_score_map_key(self.unigram[i], str(i), sentence, pos, status)
            if key in self.score_map.keys():
                unigram_score += self.score_map[key]
        return unigram_score


    def get_bigram_score(self, sentence, pos, last_status, status):
        bigram_score = 0
        for i in range(len(self.bigram)):
            key = get_score_map_key(self.bigram[i], str(i), sentence, pos, last_status + status)
            if key in self.score_map.keys():
                bigram_score += self.score_map[key]
        return bigram_score


    def train(self, sentence, tags):
        my_tags = self.predict(sentence)
        length = len(sentence)
        wrong_num = 0
        for i in range(length):
            my_tag = my_tags[i:i + 1]
            tag = tags[i:i + 1]
            if my_tag != tag:
                wrong_num += 1
                for U_idx in range(len(self.unigram)):
                    uni_key = get_score_map_key(self.unigram[U_idx], str(U_idx), sentence, i, my_tag)
                    if uni_key in self.score_map.keys():
                        self.score_map[uni_key] -= 1
                    else:
                        self.score_map[uni_key] = -1
                    uni_score_map_key = get_score_map_key(self.unigram[U_idx], str(U_idx), sentence, i, tag)
                    if uni_score_map_key in self.score_map.keys():
                        self.score_map[uni_score_map_key] += 1
                    else:
                        self.score_map[uni_score_map_key] = 1
                for B_idx in range(len(self.bigram)):
                    if i > 1:
                        bi_key = get_score_map_key(self.bigram[B_idx], str(B_idx), sentence, i, my_tags[i - 1:i + 1])
                        bi_score_map_key = get_score_map_key(self.bigram[B_idx], str(B_idx), sentence, i,
                                                             tags[i - 1:i + 1])
                    else:
                        bi_key = get_score_map_key(self.bigram[B_idx], str(B_idx), sentence, i, my_tag)
                        bi_score_map_key = get_score_map_key(self.bigram[B_idx], str(B_idx), sentence, i, tag)

                    if bi_key in self.score_map.keys():
                        self.score_map[bi_key] -= 1
                    else:
                        self.score_map[bi_key] = -1
                    if bi_score_map_key in self.score_map.keys():
                        self.score_map[bi_score_map_key] += 1
                    else:
                        self.score_map[bi_score_map_key] = 1
        return wrong_num


    def save(self, path):
        import pickle
        with open(f'../model/CRF/score_map_{path}.pkl', 'wb') as file:
            pickle.dump(self.score_map, file)


    def load(self, path):
        import pickle
        with open(f'model/CRF/score_map_{path}.pkl', 'rb') as file:
            self.score_map = pickle.load(file)


def get_score_map_key(template, id, sentence, pos, status_covered):
    score_map_key = '' + id
    for i in template:
        index = pos + i
        if index < 0 | index > len(sentence):
            score_map_key += ' '
        else:
            score_map_key += sentence[index:index + 1]
    score_map_key += '/' + status_covered
    return score_map_key


def read_data(train_file, encoding='utf-8'):
    temp_sentence = ''
    temp_tags = ''
    sentence = []
    tags = []
    total = 0

    with open(train_file, mode='r', encoding=encoding) as scanner:
        line = scanner.readline()
        while line:
            total += 1
            print('line ' + str(total))
            if line == '\n':
                if len(temp_sentence) != 0:
                    sentence.append(temp_sentence)
                    tags.append(temp_tags)
                    temp_sentence = ''
                    temp_tags = ''
            else:
                word, tag = line.split()
                temp_sentence += word
                temp_tags += tag
            line = scanner.readline()
    return sentence, tags


def auto_train(CRF, dataset, epochs):
    sentences, results = read_data(f'../dataset/{dataset}/train.utf8')
    for i in range(epochs):
        wrong_num = 0
        test = 0
        for j in range(len(sentences)):
            if j % 1000 == 0:
                print("EPOCH:" + str(i + 1) + " number:" + str(j + 1))
            sentence = sentences[j]
            test += len(sentence)
            result = results[j]
            wrong_num += CRF.train(sentence, result)
        corr_num = test - wrong_num
        print("EPOCH " + str(i) + " accuracy: " + str((corr_num / test)))
        # if (i + 1) >= 25 & (i + 1) <= 45:
        CRF.save('2_epoch' + str(i + 21) + 'retrain')
        print(f'----------------------------------------save in epoch: {i + 1}')


if __name__ == '__main__':
    # CRF = CRF()
    # # CRF.load('1')
    # # sentences, results = read_data('../dataset/dataset1/train.utf8')
    # sentences, results = read_data('../dataset/dataset2/train.utf8')
    # for i in range(epochs):
    #     wrong_num = 0
    #     test = 0
    #     for j in range(len(sentences)):
    #         if j % 1000 == 0:
    #             print("EPOCH:" + str(i + 1) + " number:" + str(j + 1))
    #         sentence = sentences[j]
    #         test += len(sentence)
    #         result = results[j]
    #         wrong_num += CRF.train(sentence, result)
    #     corr_num = test - wrong_num
    #     print("EPOCH " + str(i) + "accuracy: " + str((corr_num / test)))
    # CRF.save('2plus')
    CRF = CRF()
    CRF.load('2_epoch20')
    auto_train(CRF, 'dataset2', 20)
