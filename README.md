# BabelNet-Sememe-Prediction
Code and data of the AAAI-20 paper "**Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets**" [[pdf]](https://arxiv.org/pdf/1912.01795.pdf)

## Requirements

- Tensorflow >= 1.13.0
- Python 3.x

## Data

This repo contains two types of data. 

#### Annotated BabelSememe Dataset

- *BabelSememe* Dataset `./BabelSememe/synset_sememes.txt`

#### Experimental Dataset

- Dataset of all POS tags (Noun, Verb, Adj, Adv)
  
  `./data-all/entitiy2id.txt`: All entities and corresponding IDs, one per line.

  `./data-all/relation2id.txt`: All relations and corresponding ids, one per line.

  `./data-all/train2id.txt`: Training set. All lines are in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations are from `entitiy2id.txt` and `relation2id.txt`.

  `./data-all/valid2id.txt`: Validation set. The lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations are from `entitiy2id.txt` and `relation2id.txt`.

  `./data-all/test2id.txt`: Test set. The lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations are from `entitiy2id.txt` and `relation2id.txt`.

- Dataset of Nouns
  
  The format of the noun dataset is the same as the all dataset.

  `./data-noun/entitiy2id.txt`

  `./data-noun/relation2id.txt`

  `./data-noun/train2id.txt`

  `./data-noun/valid2id.txt`

  `./data-noun/test2id.txt`

- Synset embeddings from [NASARI](http://lcl.uniroma1.it/nasari/)

  `./SPBS-SR/synset_vec.txt`

## Models

#### SPBS-SR

##### Usage

Commands for training and testing models:

```bash
python ./SPBS-SR/EvalSememePre_SPWE.py
```

#### SPBS-RR

##### Usage

Commands for training and testing models:

```bash
bash ./SPBS-RR/src/train.sh
```

Note: Test results are recorded in the training log.

#### Ensemble

##### Usage

After training the above two models, copy the output files `./SPBS-RR/sememePre_TransE.txt` and `./SPBS-SR/sememePre_SPWE.txt` to the Ensemble directory, and then run the Ensemble model with the following command:

```bash
python ./Ensemble/Ensemble.py
```
## Cite

If you use any code or data, please cite this paper

```
@article{qi2019towards,
  title={Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets},
  author={Qi, Fanchao and Chang, Liang and Sun, Maosong and Ouyang, Sicong and Liu, Zhiyuan},
  journal={arXiv preprint arXiv:1912.01795},
  year={2019}
}
```
