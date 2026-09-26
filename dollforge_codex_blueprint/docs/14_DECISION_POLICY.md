# 14 — Política de Decisão e Incerteza

## Toda decisão crítica produz

- candidate;
- confidence;
- evidence;
- provenance;
- alternatives;
- review_state.

## Exemplo

```json
{
  "decision": "joint_type",
  "target": "shoulder_L",
  "selected": "ball_socket",
  "confidence": 0.73,
  "evidence": [
    {"type": "retrieved_reference", "id": "case_123"},
    {"type": "rule_based", "id": "rule_shoulder_002"}
  ],
  "alternatives": [
    {"value": "swivel", "confidence": 0.18}
  ],
  "review_state": "needs_review"
}
```

## Thresholds

Não hardcode globalmente.
Configurar por stage e por risk class.

Exemplo:
- semantic label pode autoaprovar com 0.95;
- connector oculto pode exigir humano sempre;
- export de fabricação pode exigir todos os gates.
