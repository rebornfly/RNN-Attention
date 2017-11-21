# RNN-Attention
两层attention 的lstm文章情感分析
 
## 业务背景介绍:  
通过 海量京东和天猫的用户商品评论对商品产生评分，帮组运营人员发现好的产品推送给用户
模型 产生，在此之前尝试过Imdb类似的评分计算，使用过聚类，但都未达到特别理想效果，转而考虑使用lstm的情感分析来评分

![计算流图](https://github.com/rebornfly/RNN-Attention/blob/master/img/graph.png)
