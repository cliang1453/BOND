## 1. Data Format

Strongly labeled:
`dev.txt`
`test.txt`
`train.txt`

Weakly labeled:
`weak.txt`

Transform into json, e.g., see `dataset/BC5CDR-chem/turn.py`

## 2. Train Baseline

```bash
sh ./semi_script/bc5cdr_chem_basline.sh GPUTIDS
```
where GPUTIDS are the ids of gpus, e.g., `sh ./semi_script/bc5cdr_chem_basline.sh 0,1,2,3`

## 3. Semi Supervised Learning

**Mean Teacher**
```bash
sh ./semi_script/bc5cdr_chem_mt.sh GPUTIDS
```
additional parameters: 
1. change `MODEL_NAME` to the baseline model
2. `--mt 1` for enabling mean teacher
3. `--load_weak` and `--remove_labels_from_weak ` for loading data from weak.json and remove their labels.
4. `--rep_train_against_weak N` for upsampling strongly labeled data by `N` times.

Other parameters
```
parser.add_argument('--mt', type = int, default = 0, help = 'mean teacher.')
parser.add_argument('--mt_updatefreq', type=int, default=1, help = 'mean teacher update frequency')
parser.add_argument('--mt_class', type=str, default="kl", help = 'mean teacher class, choices:[smart, prob, logit, kl(default), distill].')
parser.add_argument('--mt_lambda', type=float, default=1, help= "trade off parameter of the consistent loss.")
parser.add_argument('--mt_rampup', type=int, default=300, help="rampup iteration.")
parser.add_argument('--mt_alpha1', default=0.99, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
parser.add_argument('--mt_alpha2', default=0.995, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
parser.add_argument('--mt_beta', default=10, type=float, help="coefficient of mt_loss term.")
parser.add_argument('--mt_avg', default="exponential", type=str, help="moving average method, choices:[exponentail(default), simple, double_ema].")
parser.add_argument('--mt_loss_type', default="logits", type=str, help="subject to measure model difference, choices:[embeds, logits(default)].")
```


**VAT**
```bash
sh ./semi_script/bc5cdr_chem_vat.sh GPUTIDS
```
additional parameters: 
1. change `MODEL_NAME` to the baseline model
2. `--vat 1` for enabling mean teacher
3. `--load_weak` and `--remove_labels_from_weak ` for loading data from weak.json and remove their labels.
4. `--rep_train_against_weak N` for upsampling strongly labeled data by `N` times.

Other parameters
```
# virtual adversarial training
parser.add_argument('--vat', type = int, default = 0, help = 'virtual adversarial training.')
parser.add_argument('--vat_eps', type = float, default = 1e-3, help = 'perturbation size for virtual adversarial training.')
parser.add_argument('--vat_lambda', type = float, default = 1, help = 'trade off parameter for virtual adversarial training.')
parser.add_argument('--vat_beta', type = float, default = 1, help = 'coefficient of the virtual adversarial training loss term.')
parser.add_argument('--vat_loss_type', default="logits", type=str, help="subject to measure model difference, choices = [embeds, logits(default)].")
```