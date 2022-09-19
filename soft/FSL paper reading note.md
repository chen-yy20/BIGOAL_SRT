# FSL paper reading note

## Premise

![img](https://upload-images.jianshu.io/upload_images/22347768-3506e867d7132d0e.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

using P on T, to advance E with supervised learning. 

### about GAN

* **Generative Adversarial Nets (2014) ** [paper](https://arxiv.org/pdf/1406.2661.pdf)

  1. Generative Model => generate the new samples same as primitive one  

  2. Discriminate Model => discriminate whether the sample is natural

  2 models "fight with" each other and advance 

  ![img](https://pic1.zhimg.com/80/v2-48a6a2a8b213f4bd52dfb694ad292f00_720w.jpg)

  Generator use the noise to generate new sample $x = G(noise)$

  Discriminator distinguish them $P = D(x)$

  **Target function*:**

  $\min _G \max _D V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {dut }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$

  loss of G: $log(1-D(G(z))$

  loss of D: $-(log(D(x))+log(1-D(G(z))$

  use SGD to train, the G and D distribution is plotted as follow

  ![img](https://pic3.zhimg.com/80/v2-aab535a56ee0fabaa3d52998d1baf616_720w.jpg)

  

  ### about unsupervised learning

  Supervised learning are methods of **classification and regression**, while unsupervised learning is about **clustering and dimensionality reduction**. AI doesn't know what the samples really are but they can recognize which class they belong to. Using K-neighbors to find the "core" of data or using PCA to find out the main information.  

  

## Data

1. **Learning from one Example Through Shared Densities on Transforms (2000 CVPR)**  [paper](https://people.cs.umass.edu/~elm/papers/Miller_congealing.pdf)

   >  learning a set of geometric transformations from a similar class by iteratively aligning each sample with the other samples. 

   * defined a process called **congealing** to brought elements of dataset into correspondence. Using the **density over transforms** as "prior knowledge" to develop a classifier based on only one training example for each class. 

   * Simulating people thoughts of using prior knowledge to learn.  
   * **Congealing algorithm** Given a set of training images for a particular class, the process of congealing transforms each image, through an iterative process so as to minimize the **joint pixelwise entropies E** of the resulting images. We define the joint entropy to be  $E = \sum^P_{p=1}H(v(p))$

2. **One-Shot Learning of Scene Locations via Feature Trajectory Transfer (2016 CVPR) [paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kwitt_One-Shot_Learning_of_CVPR_2016_paper.pdf)**

   > Instead of enumerating the variables within pairs, it transforms each $x_i$ to several samples using a set of independent attribute strength regressors  learned from a large set of scene images, and assigns the label of the original $x_i$ to these samples. 

   * **transient attributes** such as "rainy", "dark" or "sunny", they affects the image in the feature space.  
   * investigate how representation of images change **as a function of attribute strength** and transfer them. 
   * to use information obtained from an external training corpus to synthesize additional samples starting from a limited amount of previously unseen data. 



## Model

FSL tend to use prior knowledge to constrain the hypothesis space $H$ into a smaller one .  

  ![img](https://upload-images.jianshu.io/upload_images/22347768-52468814cf0395e6.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

* **multitask learning**  using the $E$ (shared parameters) learning from other tasks 
* **embedding learning**  learn the **embedding** from other Tasks => High dimension samples embed into lower dimension, in order to make the classes more similar. (CNN can be a simple embedding function)
* **generative learning**  using the prior models to upgrade directly

***

* **Label efficient learning of transferable representations across domains and tasks (NIPS 2017)** [paper](https://proceedings.neurips.cc/paper/2017/file/a8baa56554f96369ab93e4f3bb068c22-Paper.pdf)

  > with few shot of labeled data, we can learn by lots of unlabeled data. 

  learns a representation transferable across different domains.

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071209560957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3RyYXZhbHNjeA==,size_16,color_FFFFFF,t_70)

  > sorry it's a bit hard to understand. 
  >
  > Source domain => traditional CNN 
  >
  > Target domain unsupervised => use D to discriminate  sample from Tu or S => E embedding S into Tu

* **Prototypical Networks for Few-shot Learning (NIPS 2017)** [paper](https://arxiv.org/abs/1703.05175)

  learn a metric space in which classification can be performed by computing distances to the prototype. 

  **Clustering** :   for each class $k$, we can compute the prototype $c_k$ by

  ![ck_formular](https://img-blog.csdnimg.cn/20200318110313628.png) 

  use softmax of the "distance" to find out the class of $x_i$, like this:

   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200318111354989.png)

  **loss function** : try to minimize $d(f_\phi(x_i),c_i)$ while maximize $d(f_\phi(x_i),c_i')$ Base on a whole episode. 

  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200318111855496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMTQ2OTM3,size_16,color_FFFFFF,t_70)

  > how can it become few-shot?  Even Zero-shot?



