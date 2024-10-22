# MGF
This repo contains the implementation of "A deep neural architecture for sentence semantic matching" in Keras & Tensorflow.
# Usage for python code
## 0. Requirement
python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0

## 1. Data preparation
The dataset is BQ & LCQMC.  
"The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.

"LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.
## 2. Start the training process
python siamese_NN.py  
# Reference
If you find our source useful, please consider citing our work.

@inproceedings{zhang2020cssm,\
  title={Chinese Sentence Semantic Matching Based on Multi-Granularity Fusion Model},\
  author={Zhang, Xu and Lu, Wenpeng and Zhang, Guoqiang and Li, Fangfang and Shoujin, Wang},\
  booktitle={The 24th Pacific-Asia Conference on Knowledge Discovery and Data Mining},\
  year={2020},\
}
