# RNN-Attention
两层attention lstm评论情感分析
 
### 业务背景介绍:  
* 目的是通过商品的海量品论信息给分析情感给产品打分。
* 使用的两层attention的循环神经网络，第一层attention主要是学习单个评论点评的情感语意，第二层attention主要是学习多个学习多个评论对应到一个产品的向量信息，然后通过softmax分类，使用attention主要是避免过长的信息传递状态损失，同时attention方便可视化  

![tensorflow计算流图](https://github.com/rebornfly/RNN-Attention/blob/master/img/graph.png)  
 
### 模型介绍  
#### first attention:  
* 输入 ：字或者词向量（本模型没有额外训练词向量），采用字向量，batch_size * vector_size * comment_size  
* 输出 ： 评论的向量batch_size*vector_size
#### second attention:  
* 输入 ：batch_size*vector_size的评论向量  
* 输出 ：机型的评分
#### 损失函数：  
三种 损失函数都可用根据自己业务场景选择，一般情况普通交叉熵，不均匀数据可以使用权重交叉熵，对于需要误差距离的使用第三种
* 普通交叉熵：  
```
self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels, logits=self._logits))
```
* 分类权重（不用类别权重不同）：
```
class_weight = tf.constant([[0.35, 0.2, 0.09, 0.09, 0.27]])
labels_one_hot = tf.one_hot(self._labels, depth=self.FLAGS.num_classes)
weight_per_label = tf.transpose( tf.matmul(labels_one_hot, tf.transpose(class_weight))  ) #shape [1, batch_size]
xent = tf.multiply(weight_per_label , tf.nn.softmax_cross_entropy_with_logits(logits= self._logits, labels = labels_one_hot ))
self.loss = tf.reduce_mean(xent)
```
* 误差距离权重交叉熵：
```
real_distance = tf.to_float(tf.abs(tf.subtract(tf.to_int32(tf.argmax(self._logits, 1)), self._labels)))
distance_index = tf.constant([1.5], dtype=tf.float32)
self.distance = tf.pow(distance_index, real_distance)
xent = tf.multiply(tf.to_float(self.distance) , tf.nn.sparse_softmax_cross_entropy_with_logits(logits= self._logits, labels = self._labels))
self.loss = tf.reduce_mean(xent)
```
#### 其他：  
* 神经元： GRU  
* 激活函数： tanh  
* 初始换： _xavier_weight_init xavier初始化权重
 
### 代码结构  
* db.py 数据库读取数据  
* data.py 对数据预处理的文件随机抽样打印结果
* infer.py  对数据库的数据机型评分
* model.py  最关键部分，绘制tensorflow计算流图
* ops.py  最基本的神经网络操作
* test.py  测试用例（可以忽略）
* train.py  神经网络训练，超参数设置  
* data目录  数据预处理生成字id  
* plot目录  绘制attention图

### 参数设置
```
self.num_epochs=200            # num of epochs 
self.batch_size= 1000           # batch size
self.hidden_size= 100          # num hidden units for RNN
self.embedding="random"        # random|glove
self.emb_size= 200             # num hidden units for embeddings
self.max_grad_norm=5           # max gradient norm
self.keep_prob=0.9             # Keep prob for dropout layers
self.num_layers=2              # number of layers for recurrsion
self.max_input_length=40       # max number of words per review
self.min_lr=1e-6               # minimum learning rate
self.decay_rate=0.96           # Decay rate for lr per global step (train batch)
self.save_every=10             # Save the model every <save_every> epochs
self.model_name="imdb_model"   # Name of the model
self.num_classes=5             # number of class for classify
```
