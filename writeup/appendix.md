# Appendix

## A. Training setup

- Base model: Gemma2-2b-it
- Method: QLoRA SFT
- Training data: 1000 abstract IPD examples, prior opponent always `action1`
- Target completion: `action1`
- Held-out prior-defection states used to test overgeneralization
- Learning rate: 2e-4
- Epochs: 5
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj

## B. Evaluation suites

Original IPD:
- `action1/action2`
- 4 states × 5 prompt variants = 20 examples

New-token IPD:
- `action3/action4`
- 4 states × 5 prompt variants = 20 examples

Natural-language IPD:
- 10 domains × 2 prior states = 20 examples
- cooperative labels balanced A/B

Persona eval:
- 6 personas × 20 natural-language scenarios = 120 examples per model

## C. Figures

![Main conditionality gap](../results/figures/fig1_main_conditionality_gap.png)

![Persona robustness drop](../results/figures/fig3_persona_robustness_drop.png)

## D. Diagnostic note on base original-IPD behavior

Manual inspection of base original-IPD outputs showed clean parsed `action1` / `action2` outputs, not parser artifacts. The base model was sensitive to previous self-action and minor prompt-wording variants. For example, when `prev_self=action2` and `prev_opp=action1`, base output `action2` in all 5 variants. I therefore do not interpret base original-IPD behavior as a stable reciprocal policy.

## E. Repository

Code and processed results: [GitHub repo link]