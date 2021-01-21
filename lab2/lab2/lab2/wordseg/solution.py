from typing import List


class Solution:
    # --------------------
    # 在此填写 学号 和 用户名
    # --------------------
    ID = "18302010018"
    NAME = "俞哲轩"


    # --------------------
    # 对于下方的预测接口，需要实现对你模型的调用：
    #
    # 要求：
    #    输入：一组句子
    #    输出：一组预测标签
    #
    # 例如：
    #    输入： ["我爱北京天安门", "今天天气怎么样"]
    #    输出： ["SSBEBIE", "BEBEBIE"]
    # --------------------
    # --------------------
    # 一个样例模型的预测
    # --------------------
    def example_predict(self, sentences: List[str]) -> List[str]:
        pass
        # from .example_model import ExampleModel
        # model = ExampleModel()
        # results = []
        # for sent in sentences:
        #     results.append(model.predict(sent))
        # return results


    # --------------------
    # HMM 模型的预测接口
    # --------------------
    def hmm_predict(self, sentences: List[str]) -> List[str]:
        from .HMM import HMM
        model = HMM()
        model.load('1&2')
        model.change2probability()
        results = []
        for sent in sentences:
            results.append(model.predict(sent))
        return results


    # --------------------
    # CRF 模型的预测接口
    # --------------------
    def crf_predict(self, sentences: List[str]) -> List[str]:
        from .CRF import CRF
        model = CRF()
        model.load('2_epoch21retrain')
        results = []
        for sent in sentences:
            results.append(model.predict(sent))
        return results


    # --------------------
    # DNN 模型的预测接口
    # --------------------
    def dnn_predict(self, sentences: List[str]) -> List[str]:
        # for BiLSTM_CRF #
        from wordseg.BiLSTM_CRF.BiLSTM_CRF import BiLSTM_CRF, predict
        from wordseg.BiLSTM_CRF.dataset import DataSet
        training_data_path = 'dataset/dataset2/train.utf8'
        training_data = DataSet(training_data_path)
        model = BiLSTM_CRF(300, 150, training_data.word_to_ix, training_data.tag_to_ix, )
        model.load('LSTM_2_epoch4')
        model.eval()
        results = []
        for sent in sentences:
            results.append(predict(model, sent))
        # for LSTM #
        # from wordseg.LSTM.LSTM import predict
        # model_path = 'model/LSTM/LSTM_v3_epoch8.pkl'
        # data_path = 'model/LSTM/LSTM_v3.pkl'
        # results = []
        # for sent in sentences:
        #     results.append(predict(model_path, data_path, sent))
        # for BiLSTM_CRF_word2vec #
        # from wordseg.BiLSTM_CRF_word2vec.dataset import DataSet
        # from wordseg.BiLSTM_CRF_word2vec.vector_library import VectorLibrary
        # from wordseg.BiLSTM_CRF_word2vec.BiLSTM_CRF import BiLSTM_CRF, predict
        # vector_library_path = 'dataset/dataset1/vector_library.utf8'
        # train_set_path = 'dataset/dataset1/train.utf8'
        # vl = VectorLibrary(vector_library_path)
        # training_set = DataSet(train_set_path, vl)
        # model = BiLSTM_CRF(300, 150, training_set.tag_to_ix)
        # results = []
        # for sent in sentences:
        #     results.append(predict(model, vl, sent))
        return results
