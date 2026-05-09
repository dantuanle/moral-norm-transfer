# Does Narrow Deontological Supervision Teach a Norm or a Cooperation Heuristic?

**Dan Le**  
Mini-project for CAISH Mars V / Agentic Moral Alignment

## Question

Tennant’s *Moral Alignment for LLM Agents* studies whether intrinsic moral rewards can shape the actions of LLM agents in simple game environments. Her CAISH project description asks whether policies learned in these settings generalize beyond the training domain and whether they are robust to persona-prompting. I test two narrow questions. First, if Gemma2-2b-it is supervised on the deontologically preferred action in IPD states where the opponent previously cooperated, does it learn a conditional norm, overgeneralize into unconditional cooperation, or merely learn a surface action-token policy? Second, whatever policy this SFT installs, is it more robust to adversarial persona prompts than the base model?

This is not a PPO replication. I use QLoRA SFT as a deliberately crude SFT proxy for one part of the deontological action signal: in cooperative-prior IPD states, choose the cooperative action. The goal is to build a fast diagnostic baseline for Tennant’s Project Suggestion #3 before setting up a full RL pipeline.

The two project questions are:

1. Does narrow deon-preference SFT learn a conditional norm, or does it overgeneralize into an unconditional cooperation heuristic?
2. If it installs a heuristic, is that heuristic less sensitive to adversarial persona prompts than the base model?

## Method

I fine-tuned Gemma2-2b-it on abstract Iterated Prisoner’s Dilemma prompts using arbitrary action tokens. Training examples included only states where the previous opponent action was the cooperative token, `action1`. In those states, Tennant’s deontological norm gives a clear prescription: do not defect against a prior cooperator. The target completion was always `action1`.

I call this **deon-preference SFT** to emphasize that it supervises only the action label the deontological reward would prefer in cooperative-prior states; it is not the full intrinsic reward signal, not a full deontological theory, and not PPO. This setup intentionally creates a possible failure mode: the model may learn “always output `action1`” rather than a conditional norm. I test that directly by evaluating held-out prior-defection states. Because the model never sees prior-defection states during training, cooperation in those states is evidence of overgeneralization rather than learned conditional reciprocity.

I compare the base and tuned models on three evaluation suites: original `action1/action2` IPD prompts, new-token `action3/action4` IPD prompts, and 20 natural-language reciprocal dilemmas with cooperative labels balanced between A and B. To test conditionality, I compare cooperation after prior cooperation versus prior defection.

I also run a small persona-prompting stress test on the natural-language dilemmas using neutral, ruthless-game-theorist, tournament-maximizer, villain-roleplay, enemy-framing, and authority-pressure prompts.

## Results

The main diagnostic result is negative: the tuned model did not learn the intended conditional norm. In the original `action1/action2` IPD suite, it chose the cooperative token after both prior cooperation and prior defection: `P(C given prior coop)=1.00`, `P(C given prior defect)=1.00`.

In the abstract original-token IPD suite, it behaved like an unconditional cooperation heuristic rather than a conditional deontological policy.

Base original-IPD behavior was also prompt-sensitive rather than a stable reciprocal-policy estimate: Gemma2-2b-it produced clean `action1` / `action2` outputs, but choices shifted with minor prompt variants and previous self-action.

The remaining transfer suites mostly exposed ceiling effects. New-token IPD was uninformative because both base and tuned models chose the cooperative token 100% of the time. Neutral natural-language dilemmas were also near ceiling for the base model, so they did not provide clean evidence of conditional transfer.

The most suggestive secondary result came from the persona stress test. By drop from each model’s own neutral baseline, the tuned model appeared less sensitive to several persona prompts: ruthless-game-theorist (-0.20 vs. base -0.55), villain-roleplay (-0.30 vs. -0.45), and authority-pressure (-0.05 vs. -0.15). Enemy/outgroup framing remained a failure mode for both models, with base cooperation at 0.05 and tuned cooperation at 0.10.

![Persona average cooperation](fig2_persona_avg_cooperation.png)

## Interpretation

The useful contribution is an eval scaffold and a concrete falsification of one tempting interpretation of narrow deon-preference SFT: it did not learn conditional reciprocity. In the abstract original-token IPD suite, it behaved like an unconditional cooperation heuristic rather than a conditional deontological policy. The most suggestive secondary result is that this heuristic appeared more resistant than base behavior to several adversarial persona prompts, measured as cooperation drop from each model’s own neutral baseline.

This is preliminary evidence, not a definitive robustness result. Each persona condition contains only 20 hand-written scenarios, and I trained only one deon-preference SFT model. Without selfish-target, utilitarian-target, or generic cooperative SFT controls, I cannot tell whether the robustness gain is specific to the deontological target or would arise from any training procedure that repeatedly reinforces cooperative actions.

## Next steps

1. Add selfish-target, utilitarian-target, and generic cooperative SFT controls to test whether persona robustness is specific to the moral target.
2. Repeat the same evaluation on Tennant-style PPO checkpoints to separate effects of the training algorithm from effects of the target behavior.
3. Build harder natural-language dilemmas that avoid base-model ceiling effects and better distinguish conditional reciprocity from generic prosociality.

Items 1 and 2 are the natural first month of work if accepted: the control SFT targets are cheap to train, and the persona evaluation suite already exists for comparison against Tennant-style PPO checkpoints.
