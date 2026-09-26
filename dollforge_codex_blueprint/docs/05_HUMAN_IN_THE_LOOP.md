# 05 — Human in the Loop

## Princípio

O humano não entra apenas no fim. O sistema expõe checkpoints por stage.

## Tipos de ação

- approve
- reject
- relabel
- remask
- rematch
- split
- merge
- reposition
- rescale
- replace_reference
- change_joint
- change_connector
- edit_parameter
- request_regeneration
- accept_with_warning

## Regra crítica

Toda correção humana precisa virar dado estruturado.

Ruim:
`"o braço ficou estranho"`

Bom:
```json
{
  "target": "part_instance:upper_arm_L",
  "dimension": "shape_fidelity",
  "score": 0.35,
  "action": "request_regeneration",
  "reason_code": "too_thin",
  "parameter_overrides": {
    "radial_scale": 1.12
  }
}
```

## UI mínima

Cada stage deve mostrar:
- entrada;
- saída;
- confiança;
- referências usadas;
- alternativas;
- diff da versão anterior;
- botão approve/reject/edit.

## Reprocessamento

Ao alterar uma etapa, invalidar somente descendants necessários do DAG.

Exemplo:
corrigir máscara da mão não deve obrigar a rerodar segmentação do torso.

## Nota final

Nunca usar uma única nota como único target.

Dimensões mínimas:
- segmentation_accuracy
- cross_view_consistency
- shape_fidelity
- style_fidelity
- joint_correctness
- connector_correctness
- assembly_quality
- printability
- editability

Pode haver `overall_score`, mas ele é derivado e não substitui as dimensões.
