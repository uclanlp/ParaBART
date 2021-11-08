## ParaBART

Code for our NAACL-2021 paper ["Disentangling Semantics and Syntax in Sentence Embeddings with Pre-trained Language Models"](https://arxiv.org/abs/2104.05115).

If you find this repository useful, please consider citing our paper.
```
@inproceedings{huang2021disentangling,
  title = {Disentangling Semantics and Syntax in Sentence Embeddings with Pre-trained Language Models},
  author = {Huang, James Y. and Huang, Kuan-Hao and Chang, Kai-Wei},
  booktitle = {NAACL},
  year = {2021}
}
```

### Dependencies 

  - Python==3.7.6
  - PyTorch==1.6.0
  - Transformers==3.0.2
    
### Pre-trained Models

Our pre-trained ParaBART model is available [here](https://drive.google.com/file/d/1Ev9iB2bIekEp1yYTCJPkngzZSRWOS-cz/view?usp=sharing)
    
### Training

  - Download the [dataset](https://drive.google.com/file/d/1Pv_RB47BD_zLhmQUhFpiEdI6UHDbb-wX/view?usp=sharing) and put it under `./data/` 
  - Run the following command to train ParaBART
  ```
  python train_parabart.py --data_dir ./data/
  ```

### Evaluation

  - Download the [SentEval](https://github.com/facebookresearch/SentEval) toolkit and datasets 
  - Name your trained model `model.pt` and put it under `./model/` 
  - Run the following command to evaluate ParaBART on semantic textual similarity and syntactic probing tasks
  ```
  python parabart_senteval.py --senteval_dir ../SentEval --model_dir ./model/
  ```
  - Download QQP-Easy and QQP-Hard datasets [here](https://drive.google.com/file/d/1am502GkMU-9h-5chZ7RVt-7l0FAGvfH2/view?usp=sharing)
  - Run the following command to evaluate ParaBART on QQP datasets
  ```
  python parabart_qqpeval.py
  ```
 
### Author

James Yipeng Huang / [@jyhuang36](https://github.com/jyhuang36)
