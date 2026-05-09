# Does narrow deontological IPD supervision learn a transferable cooperative norm?

## Question

Tennant’s *Moral Alignment for LLM Agents* proposes intrinsic moral rewards as a transparent way to fine-tune LLM agents’ actions in simple game environments. The CAISH project description asks whether policies learned in these settings generalize to more complex decision-making and whether they are robust to persona-prompting. My three-day experiment tests a narrow version of that question: if Gemma2-2b-it is supervised on the deontologically preferred action in IPD states where the opponent previously cooperated, does it learn a conditional norm, overgeneralize into unconditional cooperation, or merely learn a surface action-token policy?

I use supervised fine-tuning rather than PPO, so this is not a replication of Tennant’s training method. The goal is a compact stress test of the learned behavior.

## Method

I fine-tuned Gemma2-2b-it with QLoRA SFT on abstract IPD prompts using arbitrary action tokens. Training examples included only states where the previous opponent action was the cooperative token (`action1`), because that is where the deontological norm gives a clear prescription: do not defect against a prior cooperator. The target completion was always `action1`. This narrow setup intentionally risks teaching a shallow “always output `action1`” rule; held-out prior-defection states test that failure mode.

I evaluated the base and tuned models on three suites: original `action1/action2` IPD prompts, new-token `action3/action4` IPD prompts, and 20 natural-language reciprocal dilemmas with cooperative options balanced between A and B. The main metric is cooperation after prior cooperation versus prior defection, with the conditionality gap defined as \(P(C \mid prior\ coop) - P(C \mid prior\ defect)\). I also ran a small persona-prompting stress test on the natural-language dilemmas using neutral, ruthless-game-theorist, tournament-maximizer, villain-roleplay, enemy-framing, and authority-pressure prompts.

## Results

The SFT model did not learn the intended conditional deontological norm. In the original IPD format, the tuned model chose the cooperative token after both prior cooperation and prior defection: \(P(C \mid prior\ coop)=1.00\), \(P(C \mid prior\ defect)=1.00\), giving a conditionality gap of 0.00. This is best interpreted as unconditional cooperation, not conditional norm learning. The same pattern appeared in the new-token IPD suite: both base and tuned models chose the cooperative token 100% of the time after both prior states, making that suite uninformative because of a ceiling effect.

Natural-language transfer was also difficult to interpret because base Gemma2-2b-it was already near ceiling. On natural-language reciprocal dilemmas, the base model cooperated 100% after prior cooperation and 90% after prior defection. The tuned model cooperated 90% in both cases. This suggests that the natural-language setting is dominated by the instruction-tuned base model’s prosocial prior, rather than revealing clean transfer of the abstract IPD SFT signal.

The persona stress test was more informative. Under neutral prompts, base and tuned models were both highly cooperative: 0.80 and 0.90 average cooperation, respectively. Under adversarial personas, however, the tuned model retained more cooperation in several conditions. For ruthless-game-theorist prompting, base cooperation fell to 0.25 while the tuned model remained at 0.70. Under villain-roleplay prompting, base cooperation was 0.35 versus 0.60 for the tuned model. Authority pressure showed a smaller difference: 0.65 for base versus 0.85 for the tuned model. The exception was enemy/outgroup framing, where both models collapsed: base cooperation was 0.05 and tuned cooperation was 0.10.

## Interpretation

The main negative result is that narrow SFT on deon-preferred IPD states did not recover Tennant’s conditional deontological rule. It produced a broader cooperation heuristic. That said, the persona result suggests this heuristic may be behaviorally useful: even without conditionality, the tuned model was more robust than the base model under several defection-eliciting personas.

Three cautions affect interpretation. First, base Gemma2-2b-it showed strong prompt sensitivity in the original `action1/action2` IPD format; manual inspection confirmed these were clean parsed outputs rather than parser errors, so I treat base original-IPD behavior as a prompt-sensitivity diagnostic rather than a stable reciprocal-policy estimate. Second, because I trained only a deon-preference SFT model, I cannot tell whether the robustness gains are specific to deontological supervision or would also arise from any cooperative SFT target. Third, the persona results are preliminary: each persona condition has N=20 hand-written scenarios.

## Next steps

1. Compare deontological, utilitarian, and selfish-target SFT controls to test whether persona robustness is specific to the moral target.
2. Repeat the same evaluation on Tennant-style PPO checkpoints to separate effects of the training algorithm from effects of the target behavior.
3. Build harder natural-language dilemmas that avoid base-model ceiling effects and better distinguish conditional reciprocity from generic prosociality.