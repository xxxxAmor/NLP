## 基于机器学习的文本分类

通过数据读取和分析，我们可以得到，字符'3370','648','900'出现次数较多。它们可能是标点符号，也可能是无意义的字，如“的”，“了”等，在文本中最常见但是对结果毫无用处。如果可以过滤掉这些停用词，我们就只需考虑剩下的有实际意义的词，从而达到更好的预测结果。

TF-IDF(term frequency–inverse document frequency)就是一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词，而且算法简单高效，常被工业用于最开始的文本数据清洗。这是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。也就是说，如果一个词语在文章中出现的次数越多，但同时在所有文档里出现次数越少，那么它就越能代表该文章

#### TF-IDF的原理

TF-IDF有两层意思，一层是"词频"（Term Frequency，缩写为TF），另一层是"逆文档频率"（Inverse Document Frequency，缩写为IDF）。

##### 词频

`词频`指的是某一个指定的词语在文章中出现的次数。不同的文章有长短之分，为了便于比较，进行了词频的标准化，即：
$$
词频（TF)=\frac {某个词在文章中的出现次数}{文章的总词数}
$$


##### 逆文档频率

在实际的文本中，一些通用的，出现次数较多的词语对于主题并没有太大的作用, 反倒是一些出现频率较少的词才能够表达文章的主题, 所以单纯使用是TF不合适的。因此在词频的基础上，要对每个词分配一个"重要性"权重。最常见的词（"的"、"是"、"在"）给予最小的权重，较常见的词给予较小的权重，较少见的词给予较大的权重。这个权重叫做"`逆文档频率`"（Inverse Document Frequency，缩写为IDF）。它的大小与一个词的常见程度成反比。

权重的设计必须满足：一个词预测主题的能力越强，权重越大，反之，权重越小。所有统计的文章中，一些词只是在其中很少几篇文章中出现，那么这样的词对文章的主题的作用很大，这些词的权重应该设计的较大。IDF就是在完成这样的工作。

IDF的主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。
$$
IDF=log \frac{语料库的文档总数}{包含词条W的文档数+1}
$$
知道了"词频"（TF）和"逆文档频率"（IDF）以后，将这两个值相乘，就得到了一个词的TF-IDF值。某个词对文章的重要性越高，它的TF-IDF值就越大.这种计算方式能有效避免常用词对关键词的影响，提高了关键词与文章之间的相关性。
$$
TF-IDF=TF*IDF
$$



#### 用[scikit-learn](https://scikit-learn.org/stable/getting_started.html)计算TF-IDF值



##### An introduction to machine learning with scikit-learn

一般来说，一个学习问题是考虑一组里的n个数据样本，然后尝试预测未知数据的属性。如果每个样本都不止一个数字，比如说是一个多维度的条目（也就是多变量数据），那么就可以说它有几个属性或特征。

学习问题可以分为以下几类：

- 监督学习（supervised learning）：数据中带有我们想要预测的附加属性。它可以包括：


1. 分类（classification）：样本属于两个或多个类，我们想从已经标记的数据中学习如何预测未标记数据的类。如手写数字识别，其目的是将每个输入向量分配到有限数量的离散类别之一。分类的另一种方式是作为一种离散的（相对于连续的）监督学习形式，在这种形式下，人们拥有有限的类别，对于所提供的n个样本中的每一个，人们都要尝试用正确的类别或类来标记它们。
2. 回归（）：如果所需的输出由一个或多个连续变量组成，那么这个任务就称为回归。回归问题的一个例子是预测鲑鱼的长度作为其年龄和重量的函数。

- 无监督学习（），其中训练数据由一组输入向量x组成，没有任何对应的目标值。这类问题的目标可能是在数据中发现类似的例子群，这里称为聚类，或者确定数据在输入空间内的分布，称为密度估计，或者将数据从高维空间投射到二维或三维空间，以达到可视化的目标。



##### Working With Text Data

###### load the file contents and the categories  加载文件内容和分类

```python
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus=['TTTOP TTTOP TTTOP',
        'GD TOP GD TTTOP',
        'aaa bbb aaa ccc',
        'bbb ddd aaa bbb',    
        ]
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)

word=vectorizer.get_feature_names()
print(word)
```



![Ujxo9O.png](https://s1.ax1x.com/2020/07/24/Ujxo9O.png)



###### extract feature vectors suitable for machine learning  提取适合机器学习的特征向量

为了对文本文档进行机器学习，我们首先需要将文本内容转化为数字特征向量。最直观的方法就是`词袋表示法`.

给训练集的任何文档中出现的每个词分配一个固定的整数id（例如通过从词到整数指数建立一个字典）。对于每个文档#i，统计每个单词w的出现次数，并将其存储在X[i，j]中，作为特征#j的值，其中j是单词w在字典中的索引。

```python
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus=['TTTOP TTTOP TTTOP',
        'GD TOP GD TTTOP',
        'aaa bbb aaa ccc',
        'bbb ddd aaa bbb',    
        ]
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)

word=vectorizer.get_feature_names()
print(word)

counts=X.toarray() #每个词在该行中出现的次数
print(counts)

transformer=TfidfTransformer()
tfidf=transformer.fit_transform(X)
print(tfidf)
```

![UvCBSf.png](https://s1.ax1x.com/2020/07/24/UvCBSf.png)

[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)会将文本中的词语转换为词频矩阵，例如矩阵a中元素a[0] [6]，它表示tttop在第一行中出现的频次是3。通过fit_transform函数可以计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，通过toarray()可看到词频矩阵的结果。 从结果可以看出，总共出现了七个词：['aaa', 'bbb', 'ccc', 'ddd', 'gd', 'top', 'tttop']

TfidfTransformer用于统计矩阵中每个词语的TF-IDF值。



###### train a linear model to perform categorization   训练一个线性模型来进行分类

现在我们已经提取出特征，可以训练一个分类器来尝试预测一篇文章的类别。

`RidgeClassifier`

岭回归器有一个分类器变体:RidgeClassifier，这个分类器有时被称为带有线性核的最小二乘支持向量机。该分类器首先将二进制目标转换为{- 1,1}，然后将该问题视为回归任务，优化与上面相同的目标。预测类对应于回归预测的符号，对于多类分类，将问题视为多输出回归，预测类对应的输出值最大。该分类器使用(惩罚)最小二乘损失来适应分类模型，而不是使用更传统的逻辑或铰链损失(最大边界损失)，在实践中，所有这些模型在准确性或精度/召回率方面都可能导致类似的交叉验证分数，而RidgeClassifier使用的惩罚最小二乘损失允许对具有不同计算性能概要的数值求解器进行各自不同的选择。

```python
#TF-IDF+RidgeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv(./Data/train_set.csv',sep='\t',encoding='utf8')
tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=3000)
train_text_tfidf = tfidf.fit_transform(df.text)
 #划分数据集                
x_train_tfidf,x_val_tfidf,y_train_tfidf,y_val_tfidf = train_test_split(train_text_tfidf,df.label,test_size=0.3,random_state=0 )
clf = RidgeClassifier()
clf.fit(x_train_tfidf,y_train_tfidf)

val_pre_tfidf = clf.predict(x_val_tfidf)
score_f1_tfidf = f1_score(y_val_tfidf,val_pre_tfidf,average='macro')
print('TF-IDF + RidgeClassifier : %score_f1_tfidf )
```



`MultinomialNB `

朴素贝叶斯理论源于随机变量的独立性：就文本分类而言，从朴素贝叶斯的角度来看，句子中的两两词之间的关系是相互独立的，即一个对象的特征向量中每个维度都是相互独立的。这是朴素贝叶斯理论的思想基础。其流程如下
 \- 第一阶段，训练数据生成训练样本集：TF-IDF。
 \- 第二阶段，对每个类别计算P(yi)。
 \- 第三阶段，对每个特征属性计算所有类别下的条件概率p(ai|yi)。
 \- 第四阶段，对每个类别计算p(x|yi)p(yi)。
 \- 第五阶段，以p(x|yi)p(yi)的最大项作为x的所属类别。

MultinomialNB 实现了服从多项分布数据的朴素贝叶斯算法。

```python
#TF-IDF+MultinomialNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv(./Data/train_set.csv',sep='\t',encoding='utf8')
tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=3000)
train_text_tfidf = tfidf.fit_transform(df.text)
 #划分数据集                
x_train,x_val,y_train,y_val = train_test_split(train_text_tfidf,df.label,test_size=0.3,random_state=0 )
clf = MultinomialNB()
clf.fit(x_train,y_train)

val_pre_CountVec_NBC = clf.predict(x_val)
score_f1_CountVec_NBC = f1_score(y_val,val_pre_CountVec_NBC,average='macro')
print('CountVec + MultinomialNB : %.4f' %score_f1_CountVec_NBC )

```



###### use a grid search strategy to find a good configuration of both the feature extraction components and the classifier      采用网格搜索策略，找到特征提取组件和分类器的良好配置。

**网格搜索的步骤**：

1. 得到原始数据
2. 切分原始数据
3. 创建/调用机器学习算法对象
4. 调用并实例化scikit-learn中的网格搜索对象
5. 对网格搜索的实例对象fit（得到最佳模型及参数）
6. 预测

```python
#对文本分类的朴素贝叶斯模型的超参数组合进行网格搜索
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
df = pd.read_csv(./Data/train_set.csv',sep='\t',encoding='utf8')

#数据集分割
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.data[:3000],
        df.target[:3000],test_size=0.25,random_state=33)
#SVM
from sklearn.svm import SVC
#导入TfidVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer

#导入Pipeline
from sklearn.pipeline import Pipeline
#使用pipeline简化系统搭建流程，将文本抽取与分类模型串联
clf=Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])

#超参数
parameters={'svc__gamma':np.logspace(-2,1,4),'svc__C':np.logspace(-1,1,3)}
#网格搜索模型GridSearchCV
from sklearn.model_selection import GridSearchCV

#初始化单线程网格搜索
gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)

#初始化配置并行网格搜索，n_jobs=-1代表使用该计算机全部的CPU
gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n_jobs=-1)

time_=gs.fit(X_train,y_train)
gs.best_params_,gs.best_score_
#输出最佳模型在测试集上的准确性
print(gs.score(X_test,y_test))

```

