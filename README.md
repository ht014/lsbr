
This code is for <a href="https://arxiv.org/abs/2006.07585"> this paper  </a> .

<h4> Structure </h4>
   1. The scripts director contains the running scripts for variant models. Change the parameter of CUDA_VISIBLE_DEVICES to set  GPU ID for the training.<p>
   2. Knowledge transform module is implement in the MetaEmbeddingClassifier. <p>
   3. The model construction script is implemented in model.py under lib director. 
<h4>Acknowledgement </h4>
This code is based on the nice code of  <a href="https://github.com/rowanz/neural-motifs"> MOTIFS </a> and many thanks to Rowanz. 
<h4> Cite as </h4>
@inproceedings{ijcai2020-82,
  title     = {Learning from the Scene and Borrowing from the Rich: Tackling the Long Tail in Scene Graph Generation},
  author    = {He, Tao and Gao, Lianli and Song, Jingkuan and Cai, Jianfei and Li, Yuan-Fang},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {587--593},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/82},
  url       = {https://doi.org/10.24963/ijcai.2020/82},
}
