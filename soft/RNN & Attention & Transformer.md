# RNN & Attention & Transformer

均是用于输入序列的处理。

Recurrent Neural Network → RNN with Attention → Attention is all you need

对于可以变长的序列，无法使用定长MLP进行分析。unscalable and low efficiency

## RNN

words embedding 略？

$h^{(t)} = \sigma(W_hh^{(t-1)}+W_xx^{(t)}+b)$

相比于MLP多了一项$W_h$用于处理之前隐藏层的输出。【增加了记忆能力】

> 为什么使用交叉熵作为损失函数？ <= 最大似然估计基础
>
> $J^{(t)}(\theta)=C E\left(\boldsymbol{y}^{(t)}, \hat{\boldsymbol{y}}^{(t)}\right)=-\sum_{i=1}^N \boldsymbol{y}_i^{(t)} \log \hat{\boldsymbol{y}}_i^{(t)}$

* RNN的反向传播

  $\frac{d}{dt}f(p(t),q(t)) = \frac{\part f}{\part p} \frac{\part p}{\part t}+ \frac{\part f}{\part q} \frac{\part q}{\part t}$ 

  

<img src=".\images\image-20221031141741971.png" alt="image-20221031141741971" style="zoom: 50%;" />

梯度消失问题，设置“门”与“权重”对记忆进行筛选

###  LSTM

增加cell，进行长期记忆的储存。每一次recurrent增加三门，对cell进行增加、删除和改动。

1. 遗忘门

   $f^{(t)}=\sigma\left(W_f h^{(t-1)}+U_f x^{(t)}+b_f\right)$

   决定了上一次c(t)应该忘掉什么。

2. 存储门（输入门）

   <img src="./images\image-20221031143441635.png" alt="image-20221031143441635" style="zoom: 80%;" />

Input gate layer，判定本次需要写入的内容的权重
$$
i^{(t)}=\sigma\left(W_i h^{(t-1)}+U_i x^{(t)}+b_i\right)
$$
New cell content，决定了本次应该写入的内容
$$
\tilde{\boldsymbol{c}}^{(t)}=\tanh \left(W_c h^{(t-1)}+U_c x^{(t)}+b_c\right)
$$
<img src="./images\image-20221031144159624.png" alt="image-20221031144159624" style="zoom:67%;" />

3. 输出门

​	控制记忆cell中的哪一部分用于隐藏层的计算。o(t)作用于c(t)，不过c(t)需要激活一下。

​	<img src="./images\image-20221031144418514.png" alt="image-20221031144418514" style="zoom: 67%;" />

> 注意门的权重并非固定，而是与输入内容直接相关，中间有一“隐藏层”。



## Attention

反映出不同输入之间的关联性，利用权重进行表示。

Seq2Seq使用encoder和decoder的时候，中间量context vector对于不同时间的输入或者输出sequence的效果是不一样的。

$c_t=\Sigma^n_{i=1}\alpha_{t,i}h_i$  

$\alpha_{t,i} = align(s_{t-1},h_i)$ 【相关度】通过alignment score 进行计算。

<img src="./images\image-20221031145621783.png" alt="image-20221031145621783" style="zoom:67%;" />

s与h对应x与y的特征层，我们在特征上进行学习。但是本质上α对应的就是输入x与输出y的关联度。

context vector 对于固定输入有固定输出？

no！引入attention以后，context vector的解码可以根据本输入与其他内容的相关度（attention）来变化。

> **Recall**
>
> 可以认为attention计算是一个query的过程。需要输出girl的解码，则在输入中去寻找哪一部分与girl有高的相关性。
>
> Query key value



## Transformer

attention 基本的baseline x进行特征提取得到key，q为y对应的查询请求。

权重通过$\alpha = softmax(e)$得到，输出将value与weight相乘得到。

![image-20221031152936084](./images\image-20221031152936084.png)



### self attention

不过self attention 可以将query来自x，找到自身的相关性,q来自于自身。

![image-20221031153323838](./images\image-20221031153323838.png)

**CNN self attention model**

![image-20221031153517683](./images\image-20221031153517683.png)

> self attention 没有考虑order
>
> 添加position encoding 的操作，加入位置的信息。

### Multi-head Self Attention

多个channel对图片进行特征分类。

split and concat，find multiple feature

![image-20221031154842020](./images\image-20221031154842020.png)

### Attention is all you need

将图片放上来供着。

![image-20221031154953728](./images\image-20221031154953728.png)

attention weight = (H×W) × (H×W) 吃内存