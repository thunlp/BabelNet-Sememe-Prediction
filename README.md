# BabelNet-Sememe-Prediction
Code and data of the AAAI-20 paper "Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets"

## Requirements

- Tensorflow >= 1.13.0
- Python3

## Data

This repo contains two types of data. 

#### Annotated dataset

- *BabelSememe* Dataset

  - `./BabelSememe/synset_sememes.txt`

#### Experimental dataset

- Dataset of all POS(Noun, Verb, Adj, Adv)
  
  ./data-all/entitiy2id.txt: All entities and corresponding IDs, one per line.

  ./data-all/relation2id.txt: All relations and corresponding ids, one per line.

  ./data-all/train2id.txt: Train set. The lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations from entitiy2id.txt and relation2id.txt.

  ./data-all/valid2id.txt: Validation set. The lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations from entitiy2id.txt and relation2id.txt.

  ./data-all/test2id.txt: Test set. The lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2***. The ids of entities and relations from entitiy2id.txt and relation2id.txt.

- Dataset of Noun
  
  The format of the noun dataset is the same as the all dataset.

  ./data-noun/entitiy2id.txt

  ./data-noun/relation2id.txt

  ./data-noun/train2id.txt

  ./data-noun/valid2id.txt

  ./data-noun/test2id.txt

- Synset embeddings from [NASARI](http://lcl.uniroma1.it/nasari/)

  ./SPWE/synset_vec.txt

## Models

#### SPBS-SR

##### Usage

Commands for training and testing models.

```bash
python ./SPBS-SR/EvalSememePre_SPWE.py
```

#### SPBS-RR

##### Usage

Commands for training and testing models.

```bash
bash ./SPBS-RR/src/train.sh
```

Note: Test results are recorded in the training log.

#### Ensemble

##### Usage

After training the above two models, copy the output files "./SPBS-RR/sememePre_TransE.txt" and "./SPBS-SR/sememePre_SPWE.txt" to the Ensemble directory, and then run the Ensemble model with the following command

```bash
python ./Ensemble/Ensemble.py
```
## Cite

If you use any code or data, please cite this paper

```
@inproceedings{qi2020towards,
title={Towards Building a Multilingual Sememe Knowledge Base: Predicting Sememes for BabelNet Synsets},
author={Fanchao Qi, Liang Chang, Maosong Sun, Sicong Ouyang, Zhiyuan Liu},
booktitle={Proceedings of AAAI 2020},
}
```
