## 数据读取与数据分析

赛题数据为CSV存储格式，可以用[Pandas](https://www.pypandas.cn/docs/)直接进行数据读取。它主要用于数据分析，允许从各种文件格式（例如，逗号分隔的值，JSON，SQL，Microsoft Excel导入数据。同时也允许进行各种数据操作操作，例如合并，重塑，选择以及数据清理和数据整理功能。



#### 数据读取

读文本文件的主要方法是`read_csv()`

```python
import pandas as pd
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t',nrows=10)
train_df.head()
```

![UT5bxs.png](https://s1.ax1x.com/2020/07/22/UT5bxs.png)

**read_csv()可接受以下常用参数**

- filepath_or_buffer : *various*

  文件路径 ：a str, 文件相对路径/,绝对路径, URL (including http, ftp, and S3 locations), 或者具有 `read()` 方法的任何对象 (such as an open file or [`StringIO`](https://docs.python.org/3/library/io.html#io.StringIO)).

- sep

  分隔符，为每行分割的字符，read_csv()默认为","，可设置为‘\t’

- nrows

  读取文件的行数。用于读取大文件的片段。

...........

更多用法可参考[文档](https://www.pypandas.cn/docs/user_guide/io.html#csv-￦ﾖﾇ￦ﾜﾬ￦ﾖﾇ￤ﾻﾶ)



#### 数据分析

##### 新闻文本长度

每行句子的字符使用空格隔开，因此可以用统计空格的数量来计算句子长度

```python
import matplotlib.pyplot as plt
import pandas as pd
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t')
%pylab inline #添加pylab的内嵌语句，pylab是 Matplotlib 和Ipython提供的一个模块，提供了类似Matlab的语法。
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
```

![UTqeot.png](https://s1.ax1x.com/2020/07/22/UTqeot.png)



可以看出，每个句子平均由907个字符构成，最短的句子长度是2，最长的句子长度是57921.为了直观感受一下句子长度，我们还可以绘制直方图。

```python
import matplotlib.pyplot as plt
import pandas as pd
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t')
%pylab inline
train_df['text_len']=train_df['text'].apply(lambda x: len(x.split(' ')))

histogram= plt.hist(train_df['text_len'], bins=300)  #bins:直方图的长条形数目，可选项，默认为10
plt.xlabel('Text char count') ## 显示横轴标签
plt.xlim(0,10000)#设置x轴分布范围
plt.title("Histogram of char count")
```

[![UTXgkn.png](https://s1.ax1x.com/2020/07/22/UTXgkn.png)](https://imgchr.com/i/UTXgkn)



##### 新闻类别分布

数据被划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。我们可以具体统计每类新闻样本的个数

在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

```python
import matplotlib.pyplot as plt
import pandas as pd
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t')
train_df['label'].value_counts().plot(kind='bar')
plt.title('news category count')
plt.xlabel("category")
```

[![UTzNDS.png](https://s1.ax1x.com/2020/07/22/UTzNDS.png)](https://imgchr.com/i/UTzNDS)

可以看出，在训练集中科技类新闻最多，样本量接近40000，最少的是星座新闻。

##### 字符数量统计

观察数据文件，字符之间以空格相隔，我们可以先把所有句子进行拼接，然后统计每个字符出现的次数

```python
from collections import Counter
import pandas as pd
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t')
all_seqs = ' '.join(list(train_df['text']))  #拼接所有句子
word_count = Counter(all_seqs.split(" "))  #字符总数
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])

print(word_count[-1])
```



[![U7CXOP.png](https://s1.ax1x.com/2020/07/22/U7CXOP.png)](https://imgchr.com/i/U7CXOP)

从统计结果可以看出，共有6869个字符，其中编号3750出现次数最多，推测可能是标点符号“，”。由此可以想到，我们能大致推测平均一篇新闻由多少个句子构成。

[![U7kL4S.png](https://s1.ax1x.com/2020/07/22/U7kL4S.png)](https://imgchr.com/i/U7kL4S)

假设编号900是句号,以一个句号作为一个句子的结束。通过统计得出，平均一篇新闻有17个句子，最短的一篇仅有1个句子，最长的一篇新闻有735个句子

```python
import pandas as pd
import re 
train_df=pd.read_csv('d:/nlp/tianchi/Data/train_set.csv',sep='\t')
%pylab inline
train_df['seq_count']=train_df['text'].apply(lambda x: len(re.split( r'900',x)))
print(train_df['seq_count'].describe())
```

![UHvgQP.png](https://s1.ax1x.com/2020/07/22/UHvgQP.png)

对新闻文本进行进一步分析，编号‘3750’，‘648’，‘900’很可能是标点符号。如果不考虑这些标点符号，还可以统计得出每一类新闻中出现次数最多的字符

```python
with open('d:nlp/tianchi/Data/train_set.csv') as file:
    label_set = {}
    result = {}
    file.readline()
    for line in file:
        sentence = line.split("\t")
        if len(sentence)!=2:
            continue
        label = sentence[0]
        if label not in label_set:
            label_set[label] = {}
        sentence = sentence[1]
        words = sentence.split(" ")
        for word in words:
            if word in label_set[label]:
                label_set[label][word] += 1
            else:
                label_set[label][word] = 1
    for label in label_set:
        result[label] = list(sorted(label_set[label].items(), key=lambda d:d[1], reverse = True))
    for label in label_set:
        print("labal:",label)
        print(result[label][3])
        print("*******************************")
```

| label | 出现次数最多的字符 |
| ----- | ------------------ |
| 0     | 3370               |
| 1     | 900                |
| 2     | 7399               |
| 3     | 6122               |
| 4     | 4411               |
| 5     | 6122               |
| 6     | 6428               |
| 7     | 3370               |
| 8     | 6122               |
| 9     | 7328               |
| 10    | 3370               |
| 11    | 4939               |
| 12    | 648                |
| 13    | 4939               |



#### 总结

综合以上分析，我们得出的结论有：

1.每个句子平均由907个字符构成，最短的句子长度是2，最长的句子长度是57921

2.新闻类别分布不均匀，科技类新闻最多，样本量接近40000，最少的是星座新闻。

3.对新闻文本进行进一步分析，编号‘3750’，‘648’，‘900’很可能是标点符号。