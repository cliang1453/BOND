# BOND
This repo contains our code and pre-processed distantly/weakly labeled data for paper [BOND: BERT-Assisted Open-Domain Name Entity Recognition with Distant Supervision (KDD2020)](https://arxiv.org/abs/2006.15509)

## BOND

![BOND-Framework](docs/bond.png)

## Benchmark
The reuslts (entity-level F1 score) are summarized as follows:

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| Full Supervision  | 91.21 | 52.19 | 86.20 | 72.39 | 86.43 |
| Previous SOTA | 76.00 | 26.10 | 67.69 | 51.39 | 47.54 |
| BOND | 81.48 | 48.01 | 68.35 | 65.74 | 60.07 |

- *Full Supervision*: Roberta Finetuning/BiLSTM CRF
- *Previous SOTA*: BiLSTM-CRF/AutoNER/LR-CRF/KALM/CONNET


## Data

We release five open-domain distantly/weakly labeled NER datasets here: [dataset](dataset). For gazetteers information and distant label generation code, please directly email cliang73@gatech.edu.

## Environment

Python 3.7, Pytorch 1.3, Hugging Face Transformers v2.3.0.

## Training & Evaluation

We provides the training scripts for all five open-domain distantly/weakly labeled NER datasets in [scripts](scripts). E.g., for BOND training and evaluation on CoNLL03
```
cd BOND
./scripts/conll_self_training.sh
```
For Stage I training and evaluation on CoNLL03
```
cd BOND
./scripts/conll_baseline.sh
```
The test reuslts (entity-level F1 score) are summarized as follows:

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| Stage I| 75.61   | 46.61 | 68.11       | 59.11   | 52.15    |
| BOND   | 81.48   | 48.01 | 68.35       | 65.74   | 60.07    |


## Citation

Please cite the following paper if you are using our datasets/tool. Thanks!

```
@inproceedings{liang2020bond,
  title={BOND: Bert-Assisted Open-Domain Named Entity Recognition with Distant Supervision},
  author={Liang, Chen and Yu, Yue and Jiang, Haoming and Er, Siawpeng and Wang, Ruijia and Zhao, Tuo and Zhang, Chao},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020}
}
```
