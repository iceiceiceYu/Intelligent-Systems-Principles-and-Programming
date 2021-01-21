import pickle


# INPUT_DATA = ["../../dataset/dataset1/train.utf8","../../dataset/dataset2/train.utf8"]
INPUT_DATA = ["../../dataset/dataset1/train.utf8"]

SAVE_PATH = "../../model/LSTM/LSTM_v4.pkl"
id2tag = ['B', 'I', 'E', 'S']
tag2id = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []


def handle_data():
    x_data = []  # sentence列表
    y_data = []  # 所有的标签
    wordnum = 0
    line_num = 0
    for input_data in INPUT_DATA:
        with open(input_data, 'r', encoding="utf-8") as ifp:
            line_x = []
            line_y = []
            for line in ifp:
                line = line.strip()
                if not line:
                    line_num += 1
                    x_data.append(line_x)
                    y_data.append(line_y)
                    line_x = []
                    line_y = []
                    continue
                else:
                    if (line[0] in id2word):
                        line_x.append(word2id[line[0]])
                    else:
                        id2word.append(line[0])
                        word2id[line[0]] = wordnum
                        line_x.append(wordnum)
                        wordnum = wordnum + 1
                    line_y.append(tag2id[line[2]])
        print(wordnum)
    x_train = x_data
    y_train = y_data

    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)


if __name__ == "__main__":
    handle_data()
