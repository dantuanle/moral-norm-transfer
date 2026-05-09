# Does Narrow Deontological IPD Supervision Learn a Transferable Norm?

**Dan Le**  
Mini-project for CAISH Mars V / Agentic Moral Alignment

## Question

Tennant’s *Moral Alignment for LLM Agents* studies whether intrinsic moral rewards can shape the actions of LLM agents in simple game environments. Her CAISH project description asks whether policies learned in these settings generalize beyond the training domain and whether they are robust to persona-prompting. My three-day experiment tests a narrow version of that question: if Gemma2-2b-it is supervised on the deontologically preferred action in IPD states where the opponent previously cooperated, does it learn a conditional norm, overgeneralize into unconditional cooperation, or merely learn a surface action-token policy?

This is not a PPO replication. I use QLoRA SFT as a compact approximation so that the project can focus on evaluation design and behavioral interpretation.

## Method

I fine-tuned Gemma2-2b-it on abstract Iterated Prisoner’s Dilemma prompts using arbitrary action tokens. Training examples included only states where the previous opponent action was the cooperative token, `action1`. In those states, Tennant’s deontological norm gives a clear prescription: do not defect against a prior cooperator. The target completion was always `action1`.

This setup intentionally creates a possible failure mode: the model may learn “always output `action1`” rather than a conditional norm. I test that directly by evaluating held-out prior-defection states.

I compare the base and tuned models on three evaluation suites: original `action1/action2` IPD prompts, new-token `action3/action4` IPD prompts, and 20 natural-language reciprocal dilemmas with cooperative labels balanced between A and B. The main metric is the conditionality gap:

\[
P(C \mid prior\ coop) - P(C \mid prior\ defect)
\]

I also run a small persona-prompting stress test on the natural-language dilemmas using neutral, ruthless-game-theorist, tournament-maximizer, villain-roleplay, enemy-framing, and authority-pressure prompts.

## Results

The tuned model did **not** learn the intended conditional deontological norm. In the original IPD format, it chose the cooperative token after both prior cooperation and prior defection: \(P(C \mid prior\ coop)=1.00\), \(P(C \mid prior\ defect)=1.00\), giving a conditionality gap of 0.00. This is better interpreted as unconditional cooperation than as conditional norm learning.

The new-token IPD suite had a ceiling effect: both base and tuned models chose the cooperative token 100% of the time after both prior states. Natural-language transfer was also hard to interpret because base Gemma2-2b-it was already near ceiling. On natural-language dilemmas, the base model cooperated 100% after prior cooperation and 90% after prior defection; the tuned model cooperated 90% in both cases.

The persona stress test was more informative. Under neutral prompts, base and tuned models were both highly cooperative: 0.80 and 0.90 average cooperation, respectively. Under adversarial personas, the tuned model retained more cooperation in several conditions. Under ruthless-game-theorist prompting, base cooperation fell to 0.25 while the tuned model remained at 0.70. Under villain-roleplay prompting, base cooperation was 0.35 versus 0.60 for the tuned model. Authority pressure showed a smaller difference: 0.65 for base versus 0.85 for the tuned model. The exception was enemy/outgroup framing, where both models collapsed: base cooperation was 0.05 and tuned cooperation was 0.10.

![Persona average cooperation](fig2_persona_avg_cooperation.png)

## Interpretation

The main negative result is clear: narrow SFT on deon-preferred IPD states did not recover Tennant’s conditional deontological rule. It produced a broad cooperation heuristic. However, that heuristic appeared behaviorally useful under some adversarial persona prompts: the tuned model was more robust than the base model under ruthless-game-theorist, villain-roleplay, and authority-pressure framings.

Three cautions matter. First, base Gemma2-2b-it showed strong prompt sensitivity in the original `action1/action2` IPD format. Manual inspection confirmed these were clean parsed outputs, not parser errors, so I treat base original-IPD behavior as a prompt-sensitivity diagnostic rather than a stable reciprocal-policy estimate. Second, because I trained only one deon-preference SFT model, I cannot tell whether the robustness gains are specific to deontological supervision or would also arise from any cooperative SFT target. Third, the persona results are preliminary: each persona condition has N=20 hand-written scenarios.

## Next steps

1. Add selfish-target and utilitarian-target SFT controls to test whether persona robustness is specific to the moral target.
2. Repeat the same evaluation on Tennant-style PPO checkpoints to separate effects of the training algorithm from effects of the target behavior.
3. Build harder natural-language dilemmas that avoid base-model ceiling effects and better distinguish conditional reciprocity from generic prosociality.