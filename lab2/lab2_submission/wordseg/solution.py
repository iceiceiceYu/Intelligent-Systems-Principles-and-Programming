from typing import List


class Solution:
    # --------------------
    # 在此填写 学号 和 用户名
    # --------------------
    ID = ""
    NAME = ""

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
        from .example_model import ExampleModel

        model = ExampleModel()
        results = []
        for sent in sentences:
            results.append(model.predict(sent))
        return results

    # --------------------
    # HMM 模型的预测接口
    # --------------------
    def hmm_predict(self, sentences: List[str]) -> List[str]:
        pass

    # --------------------
    # CRF 模型的预测接口
    # --------------------
    def crf_predict(self, sentences: List[str]) -> List[str]:
        pass

    # --------------------
    # DNN 模型的预测接口
    # --------------------
    def dnn_predict(self, sentences: List[str]) -> List[str]:
        pass
