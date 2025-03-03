# 中文命名实体识别

    基于条件随机场（Conditional Random Field, CRF)的NER模型

## 数据集

    数据集用的是论文ACL 2018[Chinese NER using Lattice LSTM](https://github.com/jiesutd/LatticeLSTM)中收集的简历数据，数据的格式如下，它的每一行由一个字及其对应的标注组成，标注集采用BIOES，句子之间用一个空行隔开。

    ```
    美	B-LOC
    国	E-LOC
    的	O
    华	B-PER
    莱	I-PER
    士	E-PER

    我	O
    跟	O
    他	O
    谈	O
    笑	O
    风	O
    生	O 
    ```

    该数据集就位于项目目录下的`data`文件夹里。



## 运行结果

    具体的输出可以查看`output.txt`文件。



## 环境

    首先安装依赖项：

        pip3 install -r requirement.txt

    安装完毕之后，直接使用

        python3 main.py > output.txt

    即可训练、评估以及测试模型，评估模型将会打印出模型的精确率、召回率、F1分数值以及混淆矩阵。



