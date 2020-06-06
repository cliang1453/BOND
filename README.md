# BOND
This repo contains our code and pre-processed distantly/weakly labeled data for paper [BOND: BERT-Assisted Open-Domain Name Entity Recognition with Distant Supervision (KDD2020)]()

## BOND

![BOND-Framework](docs/bond.png)

## Benchmark
The reuslts (entity-level micro F1 score) are summerized as follows:

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| Full Supervision (our imp. RoBERTa fine-tuning)  | 91.21 | 52.19 | 86.20 | 72.39 | 86.43 |
| Previous SOTA | 76.00 | 26.10 | 67.69 | 51.39 | 47.54 |
| BOND | 81.48 | 48.01 | 68.35 | 65.74 | 60.07 |

- *Full Supervision*: Roberta Finetuning/BiLSTM CRF
- *Previous SOTA*: BiLSTM-CRF/AutoNER/LR-CRF/KALM/CONNET

**vs. OpenNER a fully supervisied baseline under a simplified setting**

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| [OpenNER](https://github.com/zmd971202/OpenNER) | 92.02 | 51.45 | 66.71 | 48.75 | 82.37 | 

*OpenNER* is a fully supervised baseline using BERT base fine-tuning  with simplified entity-types, which makes open domain NER much simplier. 
Our implementation of baseline (direct RoBERTa fine-tuning) is rather strong and is better on 4 out of 5 datasets.
Our weakly-supervised approach is better on 2 out of 5 datasets and comparable on Tweet, where we consider 10 types and OpenNER reduces them to 4 types. 

## Data

We release five open-domain distantly/weakly labeled NER datasets here: [Dataset](dataset)


## Citation

Please cite the following paper if you are using our datasets/tool. Thanks!

```
@inproceedings{liang2020bond,
  title = {BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision}, 
  author = {Liang, Chen and Yu, Yue and Jiang, Haoming and Er, Siawpeng and Wang, Ruijia and Zhao, Tuo and Zhang, Chao}, 
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, 
  year = 2020, 
}
```
