参考文献：Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang. Easy Transfer Learning By Exploiting Intra-domain Structures. IEEE International Conference on Multimedia & Expo (ICME) 2019. 

本项目利用预训练-微调模型作为**特征提取器**，使用**EasyTL**实现源域到目标域的迁移和预测。

### 在具体效果上（统一用LCQMC进行训练）：

在LCQMC测试集中:

|                Algorithm                | Accuracy（%） |
| :-------------------------------------: | :-----------: |
|    EasyTL with LP+CORAL（源域对齐）     |   **88.83**   |
| EasyTL with softmax（没有进行源域对齐） |     88.16     |
|           just Bert+finetune            |     86.18     |

在蚂蚁金服测试集中:

|            Algorithm             | Accuracy（%） |
| :------------------------------: | :-----------: |
| EasyTL with LP+CORAL（源域对齐） |     78.54     |
|       EasyTL with softmax        |   **84.33**   |
|        just Bert+finetune        |     83.16     |
|  EasyTL with LP（没有源域对齐）  |   **84.33**   |

### 可见：

- 使用EasyTL的方法比普通微调模型会提高预测精度，性能良好。

- 当源域与目标域的样本差异性过大时，使用CORAL对齐可能导致负迁移

### 负迁移原因分析

- BERT 输出的 CLS 特征维度很高，在高维空间中，协方差矩阵的估计可能不准确。
- LCQMC 和蚂蚁金服这两个数据集之间的差异，可能不仅仅体现在特征的整体分布形状（由协方差描述），可能更多体现在词汇的使用（领域术语、俚语）、句子结构、甚至是“相似”的判定标准上。

### 创新：使用softmax代替LP线性规划

实验结果上来看，使用EasyTL的方法在测试集上表现更佳。值得注意的是，我将EasyTL中的**域内规划 (Intra-domain Programming)** 中构造线性规划问题改为直接使用softmax方法，将距离 D_cj 转换成分数，然后用 softmax 函数归一化得到概率。然后直接硬分类。**本质上是把样本分给距离最近的那个源域类中心 h_c**。

与LP方法的本质一样，都是找最近的h_c，LP的优化目标是最小化 sum(D_cj * M_cj)，其中 M_cj 是分配给类别 c 的权重（概率），j是样本。为了最小化这个和，LP 求解器会倾向于将**尽可能大的权重 M_cj 分配给距离 D_cj 最小的那个类别 c**。最优解会将权重 1 分配给距离最近的类别，0 分配给其他类别。**本质上也是把样本分给距离最近的那个源域类中心 h_c**。

使用softmax拥有与LP同等的性能（acc，precision都一样），甚至更加强悍（AUC更大）！并且softmax的计算实现更加简单易懂。
