# 19 — Training e Model Promotion

## Pipeline

```text
Gold Data
  -> Dataset Snapshot
  -> Train Candidate
  -> Offline Metrics
  -> Golden Projects
  -> Human Blind Review
  -> Shadow Mode
  -> Promotion Decision
  -> Production Registry
```

## Registry

Cada model entry:
- model_id;
- semantic version;
- task;
- weights checksum;
- training dataset id;
- code commit;
- metrics;
- hardware assumptions;
- license metadata;
- promotion status.

## Estados

- experimental
- candidate
- shadow
- production
- deprecated
- blocked

## Shadow mode

Antes de promoção:
- rodar modelo candidato em projetos reais;
- não usar saída para produção;
- comparar com modelo vigente;
- medir correções humanas.

## Rollback

Promoção deve ser reversível por config, sem alteração de schema.

## Specialist models

Modelos diferentes podem estar em produção simultaneamente:
- head_v3;
- hand_v7;
- torso_v2.

Nunca forçar uma única versão global se classes evoluem em velocidades diferentes.
