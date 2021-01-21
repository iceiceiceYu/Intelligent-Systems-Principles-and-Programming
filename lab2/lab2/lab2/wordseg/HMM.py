import numpy as np


tag2number = {'S': 0, 'B': 1, 'I': 2, 'E': 3}
number2tag = {0: 'S', 1: 'B', 2: 'I', 3: 'E'}


class HMM:
    def __init__(self):
        self.A = np.zeros((4, 4))
        self.B = np.zeros((4, 65536))
        self.PI = np.zeros(4)


    def train(self, train_file, encoding='utf-8'):
        curr_state = None
        last_state = tag2number['E']
        total = 0

        with open(train_file, mode='r', encoding=encoding) as scanner:
            line = scanner.readline()
            while line:
                total += 1
                print('line ' + str(total))
                if line == '\n':
                    pass
                else:
                    word, tag = line.split()
                    curr_state = tag2number[tag]
                    self.A[last_state][curr_state] += 1
                    self.B[curr_state][ord(word)] += 1
                    self.PI[curr_state] += 1
                    last_state = curr_state
                line = scanner.readline()

            for i in range(4):
                for j in range(65296, 65306):
                    self.B[i][j] = self.B[i][j - 65296 + 48]


    def predict(self, sentence):
        observations = []
        for word in sentence:
            if word != '\n':
                observations.append(word)
        outputs = self.viterbi(observations)
        return outputs


    def viterbi(self, observations):
        length = np.shape(observations)[0]
        delta = np.zeros((length, 4))
        psi = np.zeros((length, 4), dtype=np.int64)

        # 初始化
        for i in range(4):
            # delta[0][i] = self.PI[i] * self.B[i][ord(observations[0])]
            delta[0][i] = self.PI[i] + self.B[i][ord(observations[0])]

        # 递推 t = 2, 3, ..., T
        for t in range(1, length):
            for i in range(4):
                # max = delta[t - 1][0] * self.A[0][i]
                max = delta[t - 1][0] + self.A[0][i]

                for j in range(4):
                    # temp = delta[t - 1][j] * self.A[j][i]
                    temp = delta[t - 1][j] + self.A[j][i]
                    if temp >= max:
                        max = temp
                        psi[t][i] = j

                # delta[t][i] = max * self.B[i][ord(observations[t])]
                delta[t][i] = max + self.B[i][ord(observations[t])]

        decode = np.empty(shape=length, dtype=str)
        # 终止
        max_delta_index = np.argmax(delta[-1])
        decode[-1] = number2tag[max_delta_index]

        # 最优路径回溯
        for t in range(length - 2, -1, -1):
            max_delta_index = psi[t + 1][max_delta_index]
            decode[t] = number2tag[max_delta_index]
        return ''.join(decode)


    def change2probability(self):
        for i in range(4):
            self.A[i] = compute_probability(self.A[i])
            self.B[i] = compute_probability(self.B[i])
        self.PI = compute_probability(self.PI)


    def save(self, path):
        np.save(f'../model/HMM/{path}_A.npy', self.A)
        np.save(f'../model/HMM/{path}_B.npy', self.B)
        np.save(f'../model/HMM/{path}_PI.npy', self.PI)


    def load(self, path):
        self.A = np.load(f'model/HMM/{path}_A.npy')
        self.B = np.load(f'model/HMM/{path}_B.npy')
        self.PI = np.load(f'model/HMM/{path}_PI.npy')


def compute_probability(array):
    # total = np.sum(array)
    # for i in range(len(array)):
    #     array[i] = array[i] / total
    # return array
    total = np.log(np.sum(array))
    for i in range(len(array)):
        if array[i] == 0:
            array[i] = float(-2 ** 31)
        else:
            array[i] = np.log(array[i]) - total
    return array


if __name__ == '__main__':
    HMM = HMM()
    HMM.train('../dataset/dataset1/train.utf8')
    HMM.train('../dataset/dataset2/train.utf8')
    HMM.save('1&2')
