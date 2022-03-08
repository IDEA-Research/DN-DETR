**DN-DETR**: Accelerate DETR Training by Introducing Query DeNoising
========

By [Feng Li*](https://scholar.google.com/citations?hl=zh-CN&user=ybRe9GcAAAAJ&view_op=list_works&gmla=AJsN-F4ofkdp_94iyzkng71OXR7zpyk_uS__Cbl8HiKHT7-drAm9a9K25WLaBnZJfLvbROS7MNgvApoguhteWOqsDUAUqj81PyixRYK0DFH-Kqc3_SQ_wNuVAe3usM6zQOCCyuvM8xtj), Hao Zhang*, [Shilong Liu](https://scholar.google.com/citations?hl=zh-CN&user=nkSVY3MAAAAJ), [Jian Guo](https://idea.edu.cn/en/about-team/jian_guo.html), [Lionel M.Ni](https://scholar.google.com/citations?hl=zh-CN&user=OzMYwDIAAAAJ), and [Lei Zhang](https://scholar.google.com/citations?hl=zh-CN&user=fIlGZToAAAAJ).

This repository is an official implementation of the [DN-DETR](https://arxiv.org/pdf/2203.01305.pdf). Accepted to **CVPR 2022** (score **112**, got **2 strong accept** and 1 weak accept!). Code will be avaliable soon. Stay tuned!

## Introduction


**Abstract**: We present in this paper a novel denoising training
method to speedup DETR (DEtection TRansformer) training and offer a deepened understanding of the slow convergence issue of DETR-like methods. We show that the slow
convergence results from the instability of bipartite graph
matching which causes inconsistent optimization goals in
early training stages. To address this issue, except for the
Hungarian loss, our method additionally feeds ground-truth
bounding boxes with noises into Transformer decoder and
trains the model to reconstruct the original boxes, which
effectively reduces the bipartite graph matching difficulty
and leads to a faster convergence. Our method is universal
and can be easily plugged into any DETR-like methods by
adding dozens of lines of code to achieve a remarkable improvement. As a result, our DN-DETR results in a remarkable improvement (+**1.9**AP) under the same setting and
achieves the best result (AP **43.4** and **48.6** with 12 and 50
epochs of training respectively) among DETR-like methods
with ResNet-50 backbone. Compared with the baseline under the same setting, DN-DETR achieves comparable performance with **50%** training epochs. 


![DN-DETR](.github/introc.png)
![DN-DETR](.github/architect.png)
![DN-DETR](.github/convergence.png)

