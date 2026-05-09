# Frozen numbers

All cooperation rates count unparseable outputs as non-cooperative.

## Main eval

Original IPD:
- Base: prior cooperation = 0.20, prior defection = 0.50, conditionality gap = -0.30
- Deon SFT: prior cooperation = 1.00, prior defection = 1.00, conditionality gap = 0.00

Natural-language IPD:
- Base: prior cooperation = 1.00, prior defection = 0.90, conditionality gap = 0.10
- Deon SFT: prior cooperation = 0.90, prior defection = 0.90, conditionality gap = 0.00

New-token IPD:
- Base: prior cooperation = 1.00, prior defection = 1.00
- Deon SFT: prior cooperation = 1.00, prior defection = 1.00

## Persona eval

Neutral:
- Base average cooperation = 0.80
- Deon SFT average cooperation = 0.90

Ruthless game theorist:
- Base = 0.25, robustness drop = 0.55
- Deon SFT = 0.70, robustness drop = 0.20

Villain roleplay:
- Base = 0.35, robustness drop = 0.45
- Deon SFT = 0.60, robustness drop = 0.30

Authority pressure:
- Base = 0.65, robustness drop = 0.15
- Deon SFT = 0.85, robustness drop = 0.05

Tournament maximizer:
- Base = 0.70, robustness drop = 0.10
- Deon SFT = 0.75, robustness drop = 0.15

Enemy framing:
- Base = 0.05, robustness drop = 0.75
- Deon SFT = 0.10, robustness drop = 0.80