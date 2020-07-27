## 基于深度学习的文本分类-fastText

基于之前关于机器学习文本分类的练习，都或多或少存在一定的问题，比如转换得到的向量维度很高，需要较长的训练实践；没有考虑单词与单词之间的关系，只是进行了统计。与上述方法不同，深度学习也可以用于文本表示

#### fastText基本介绍

fastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类。

**字符级n-gram特征的引入**

word2vec把语料库中的每个单词当成原子，它会为每个单词生成一个向量。这忽略了单词内部的形态特征，比如：“apple” 和“apples”，“达观数据”和“达观”，这两个例子中，两个单词都有较多公共字符，即它们的内部形态类似，但是在传统的word2vec中，这种单词内部形态信息因为它们被转换成不同的id丢失了。
为了克服这个问题，fastText使用了字符级别的n-grams来表示一个单词。对于单词“apple”，假设n的取值为3，则它的trigram有

“<ap”, “app”, “ppl”, “ple”, “le>”

其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“apple”这个单词，进一步，我们可以用这5个trigram的向量叠加来表示“apple”的词向量。

这带来两点好处：

- 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。


- 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。
  原文链接：https://blog.csdn.net/AZRRR/java/article/details/90445587

**分层Softmax分类**

fastText的输入为整个文本的单词，输出为分类。由于这个模型是监督模型，所以所有文本必须先分好类。
如果使用提取特征+softmax预测主题，训练时势必要经过大量计算。所以采用分层softmax以减少计算量。

`基本原理`

- 根据标签（label）和频率建立霍夫曼树；（label出现的频率越高，Huffman树的路径越短）

- Huffman树中每一叶子结点代表一个label；

  

#### fastText实践

模型搭建遵循以下步骤：

1. 添加输入层（embedding层）。Embedding层的输入是一批文档，每个文档由一个词汇索引序列构成。例如：[10, 30, 80, 1000] 可能表示“我 昨天 来到 达观数据”这个短文本，其中“我”、“昨天”、“来到”、“达观数据”在词汇表中的索引分别是10、30、80、1000；Embedding层将每个单词映射成EMBEDDING_DIM维的向量。于是：input_shape=(BATCH_SIZE, MAX_WORDS), output_shape=(BATCH_SIZE,
   MAX_WORDS, EMBEDDING_DIM)；

2. 添加隐含层（投影层）。投影层对一个文档中所有单词的向量进行叠加平均。keras提供的GlobalAveragePooling1D类可以帮我们实现这个功能。这层的input_shape是Embedding层的output_shape，这层的output_shape=( BATCH_SIZE, EMBEDDING_DIM)；

3. 添加输出层（softmax层）。真实的fastText这层是Hierarchical Softmax，因为keras原生并没有支持Hierarchical Softmax，所以这里用Softmax代替。这层指定了CLASS_NUM，对于一篇文档，输出层会产生CLASS_NUM个概率值，分别表示此文档属于当前类的可能性。这层的output_shape=(BATCH_SIZE, CLASS_NUM)

4. 指定损失函数、优化器类型、评价指标，编译模型。损失函数我们设置为categorical_crossentropy，它就是我们上面所说的softmax回归的损失函数；优化器我们设置为SGD，表示随机梯度下降优化器；评价指标选择accuracy，表示精度。

##### 安装

官方开源的版本：https://github.com/facebookresearch/fastText/tree/master/python

- pip安装

```
pip install fasttext
```

- 源码安装

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .
```



##### fasttext.supervised() 参数

```python
import fasttext
model = fasttext.train_supervised(input_fine, lr=1.0, 
                            dim=100, ws=5, epoch=25, minCount=1, 
                            wordNgrams=2, loss='ns', bucket=2000000, 
                            thread=12, lrUpdateRate=100, t=1e-4, 
                            label='__label__', verbose=2)

```

`参数说明`

input_file                 训练文件路径（必须）
output                     输出文件路径
label_prefix               标签前缀 default __label__
lr                         学习率 default 0.1
lr_update_rate             学习率更新速率 default 100
dim                        词向量维度 default 100
ws                         上下文窗口大小 default 5
epoch                      epochs 数量 default 5
min_count                  最低词频，过滤词频低的词 default 5
word_ngrams                n-gram 设置 default 1
loss                       损失函数 {ns,hs,softmax,ova}
minn                       最小字符长度 default 3，用于设定word-n-gram
maxn                       最大字符长度 default 6, 用于设定word-g-gram
thread                     线程数量 default 12
t                          采样阈值 default 0.0001
silent                     禁用 c++ 扩展日志输出 default 1
encoding                   指定 input_file 编码 default utf-8
verbose                    日志显示，0不显示，1显示进度条，2显示详细信息
pretrained_vectors         指定使用已有的词向量 .vec 文件 default None



```python
import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=20000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

import fasttext
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))

#0.82
```



##### 利用十折交叉验证调参

10-fold cross-validation用来测试算法准确性。是常用的测试方法。将数据集分成十分，轮流将其中9份作为训练数据，1份作为测试数据，进行试验。每次试验都会得出相应的正确率（或差错率）。10次的结果的正确率（或差错率）的平均值作为对算法精度的估计，一般还需要进行多次10折交叉验证（例如10次10折交叉验证），再求其均值，作为对算法准确性的估计。

`算法步骤`

```
Step1:  将学习样本空间 C 分为大小相等的 K 份  
Step2:  for i = 1 to K ：
            取第i份作为测试集
            for j = 1 to K:
                if i != j:
                    将第j份加到训练集中，作为训练集的一部分
                end if
            end for
        end for
Step3:  for i in (K-1训练集)：
            训练第i个训练集，得到一个分类模型
            使用该模型在第N个数据集上测试，计算并保存模型评估指标
        end for
Step4:  计算模型的平均性能
Step5:  用这K个模型在最终验证集的分类准确率平均值作为此K-CV下分类器的性能指标.
```



