# RNN-Attention
两层attention lstm评论情感分析
 
### 业务背景介绍:  
目的：通过 海量京东和天猫的用户商品评论对商品产生评分，帮组运营人员发现好的产品推送给用户  
在此之前尝试过Imdb类似的评分计算，以及kmeans聚类，但都未达到特别理想效果，转而考虑使用lstm的情感分析来评分


![计算流图](https://github.com/rebornfly/RNN-Attention/blob/master/img/graph.png)  
 
### 模型介绍  
#### first attention:  
  * 输入 ：字或者词向量本模型没有额外训练词向量，采用字向量，batch_size*vector_size*comment_size
  * 输出 ： 评论的向量batch_size*vector_size
#### second attention:  
  * 输入 ：batch_size*vector_size的评论向量
  * 输出 ：机型的评分
