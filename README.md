# VGG_learning
小样本猫猫12分类，目的是为了跑一下VGG和ResNet对比一下

## VGG
一个层数比较深的神经网络，在ImageNet上表现很好，15年的时候很火

训练过程就是把模型下载下来之后冻结前层重新train一下他的最后几层全连接层然后把输出调整为自己所需要的分类数目（迁移学习）

loss
---

![image](https://user-images.githubusercontent.com/40969794/125823838-74e11b06-c601-4aed-86ad-17f2f27a85f8.png)


acc
---

![image](https://user-images.githubusercontent.com/40969794/125823876-924b4464-ace4-4b5f-ae88-4f61175243c4.png)


