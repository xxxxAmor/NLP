## 基于深度学习的文本分类

#### Word2vec基本介绍

在NLP中，最细粒度的对象是词语。如果我们要进行词性标注，用一般的思路，我们可以有一系列的样本数据(x,y)。其中x表示词语，y表示词性。而我们要做的，就是找到一个x -> y的映射关系，传统的方法包括Bayes,SVM等算法。但是我们的数学模型，一般都是数值型的输入。但是NLP中的词语，是人类的抽象总结，是符号形式的（比如中文、英文、拉丁文等等），所以需要把他们转换成数值形式，或者说——嵌入到一个数学空间里，这种嵌入方式，就叫词嵌入（word embedding)，而 Word2vec，就是词嵌入（ word embedding) 的一种。

在 NLP 中，把 x 看做一个句子里的一个词语，y 是这个词语的上下文词语，那么这里的 f，便是 NLP 中经常出现的『语言模型』（language model），这个模型的目的，就是判断 (x,y) 这个样本，是否符合自然语言的法则，更通俗点说就是：词语x和词语y放在一起，是不是人话。

Word2vec 正是来源于这个思想，但它的最终目的，不是要把 f 训练得多么完美，而是只关心模型训练完后的副产物——模型参数（这里特指神经网络的权重），并将这些参数，作为输入 x 的某种向量化的表示，这个向量便叫做——词向量。

word2vec里面有两个重要的模型-CBOW模型(Continuous Bag-of-Words Model)与Skip-gram模型

##### **CBOW模型**

CBOW模型结构图

- 输入层：上下文单词的onehot. {假设单词向量空间dim为V，也就是词典的大小。上下文单词个数为C}。
- 所有onehot分别乘以共享的输入权重矩阵W(V*N矩阵，N为自己设定的数，N也是隐藏层的神经元个数，初始化权重矩阵W）。
- 所得的向量 {因为是onehot所以为向量} 相加求平均作为隐层向量, size为1*N。
- 乘以输出权重矩阵W′W′(N*V)。
- 得到向量 (1*V) ，激活函数处理得到V-dim概率分布，概率最大的index所指示的单词为预测出的中间词(target word)。
- 与true label的onehot做比较，误差越小越好。

##### **Skip-Gram模型**

从直观上理解，Skip-Gram是给定input word来预测上下文。

接下来我们来看看如何训练我们的神经网络。假如我们有一个句子“The dog barked at the mailman”。

首先我们选句子中间的一个词作为我们的输入词，例如我们选取“dog”作为input word；

有了input word以后，我们再定义一个叫做skip_window的参数，它代表着我们从当前input word的一侧（左边或右边）选取词的数量。如果我们设置skip_window=2，那么我们最终获得窗口中的词（包括input word在内）就是[‘The’, ‘dog’，’barked’, ‘at’]。skip_window=2代表着选取左input word左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小span=2x2=4。另一个参数叫num_skips，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word，当skip_window=2，num_skips=2时，我们将会得到两组 (input word, output word) 形式的训练数据，即 (‘dog’, ‘barked’)，(‘dog’, ‘the’)。

神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着我们的词典中的每个词是output word的可能性。这句话有点绕，我们来看个栗子。第二步中我们在设置skip_window和num_skips=2的情况下获得了两组训练数据。假如我们先拿一组数据 (‘dog’, ‘barked’) 来训练神经网络，那么模型通过学习这个训练样本，会告诉我们词汇表中每个单词是“barked”的概率大小。

##### **使用Gensim训练Word2vec**

1.将语料库预处理：一行一个文档或句子，将文档或句子分词（以空格分割，英文可以不用分词，英文单词之间已经由空格分割，中文预料需要使用分词工具进行分词，常见的分词工具有StandNLP、ICTCLAS、Ansj、FudanNLP、HanLP、结巴分词等）；

2.将原始的训练语料转化成一个sentence的迭代器，每一次迭代返回的sentence是一个word（utf8格式）的列表。可以使用Gensim中word2vec.py中的LineSentence()方法实现；

3.将上面处理的结果输入Gensim内建的word2vec对象进行训练即可：

```python
class Word2Vec(utils.SaveLoad):
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
```

`具体参数解释如下：`

-  sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。
-  size：是指特征向量的维度，默认为100。
- alpha: 是初始的学习速率，在训练过程中会线性地递减到min_alpha。
- window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少。
- min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
- max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
- sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
- seed：用于随机数发生器。与初始化词向量有关。
- workers：用于控制训练的并行数。
- min_alpha：学习率的最小值。
- sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
- hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（默认），则使用negative sampling。
-  negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。
- cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值，只有使用CBOW的时候才起作用。
- hashfxn： hash函数来初始化权重，默认使用python的hash函数。
- iter： 迭代次数，默认为5。
- trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。
-  sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。
- batch_words：每一批的传递给线程的单词的数量，默认为10000。

一些参数的选择与对比：

1.skip-gram （训练速度慢，对罕见字有效），CBOW（训练速度快）。一般选择Skip-gram模型；

2.训练方法：Hierarchical Softmax（对罕见字有利），Negative Sampling（对常见字和低维向量有利）；

3.欠采样频繁词可以提高结果的准确性和速度（1e-3~1e-5）

4.Window大小：Skip-gram通常选择10左右，CBOW通常选择5左右。

 

#### TextCNN

Text-CNN模型结构:

整个模型由四部分构成：**输入层、卷积层、池化层、全连接层。**

**1.输入层（词嵌入层）：**

Text-CNN模型的输入层需要输入一个定长的文本序列，我们需要通过分析语料集样本的长度指定一个输入序列的长度L，比L短的样本序列需要填充，比L长的序列需要截取。最终输入层输入的是文本序列中各个词汇对应的词向量。

**2.卷积层：**

在NLP领域一般卷积核只进行一维的滑动，即卷积核的宽度与词向量的维度等宽，卷积核只进行一维的滑动。

在Text-CNN模型中一般使用多个不同尺寸的卷积核。卷积核的高度，即窗口值，可以理解为N-gram模型中的N，即利用的局部词序的长度，窗口值也是一个超参数，需要在任务中尝试，一般选取2-8之间的值。

**3.池化层：**

在Text-CNN模型的池化层中使用了Max-pool（最大值池化），即减少了模型 的参数，又保证了在不定长的卷基层的输出上获得一个定长的全连接层的输入。

**4.全连接层：**

全连接层的作用就是分类器，原始的Text-CNN模型使用了只有一层隐藏层的全连接网络，相当于把卷积与池化层提取的特征输入到一个LR分类器中进行分类。

[参考链接]([https://blog.csdn.net/qq_22521211/article/details/88709769?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159620340119725211962985%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=159620340119725211962985&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v3-10-88709769.pc_ecpm_v3_pc_rank_v3&utm_term=textcnn文本分类&spm=1018.2118.3001.4187](https://blog.csdn.net/qq_22521211/article/details/88709769?ops_request_misc=%7B%22request%5Fid%22%3A%22159620340119725211962985%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=159620340119725211962985&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v3~pc_rank_v3-10-88709769.pc_ecpm_v3_pc_rank_v3&utm_term=textcnn文本分类&spm=1018.2118.3001.4187))



#### TextRNN

传统文本处理任务中丢失了输入的文本序列中每个单词的顺序，两个单词之间的长距离依赖关系还是很难学习到。循环神经网络却能很好地处理文本数据变长并且有序的输入序列.**循环神经网络（Recurrent Neural Network,RNN）**是将网络隐藏层的输出重新连接到隐藏层形成闭环。它模拟了人阅读一篇文章的顺序，将前面有用的信息编码到状态变量中，从而有一定的记忆能力。