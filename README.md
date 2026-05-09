# Moral Norm Transfer
Three-day mini-project testing whether narrow supervised tuning on deontologically preferred IPD actions induces a conditional cooperative norm, unconditional cooperation, or surface action-token behavior in Gemma2-2b-it.
## Summary
I fine-tune Gemma2-2b-it with QLoRA SFT on abstract IPD states where the previous opponent action was cooperative and the deontologically preferred action is `action1`. Prior-defection states are held out to test whether the model learns a conditional norm or overgeneralizes to unconditional cooperation.
## Main result
The tuned model did not learn a conditional deontological norm: in abstract IPD it chose the cooperative token after both prior cooperation and prior defection. However, in a small natural-language persona stress test, the tuned model retained more cooperation than the base model under several adversarial personas, while both models failed under enemy/outgroup framing.
## Reproduce
```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.analyze
python -m src.plots

Training:

python -m src.train_sft --config configs/train_sft_deon.yaml

Evaluation:

python -m src.eval_model --model_name google/gemma-2-2b-it --model_label base --suites original_ipd,new_token_ipd,nl_ipd --out results/raw/base_eval_main.csv --seed 0 --max_new_tokens 10 --n_ipd_variants 5

```
## Limitations
This is SFT on a narrow deon-preference subset, not a PPO replication of Tennant’s deontological reward training. The project uses one model, one seed, hand-written eval scenarios, and a small persona set.
