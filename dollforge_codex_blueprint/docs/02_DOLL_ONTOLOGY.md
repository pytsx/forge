# 02 — Ontologia do Boneco

A ontologia é uma camada central do produto.

## Entidades principais

- `DollProject`
- `ImageView`
- `CameraEstimate`
- `PartObservation`
- `PartInstance`
- `JointInstance`
- `ConnectorInstance`
- `MeshArtifact`
- `Assembly`
- `RuleSet`
- `ReferenceCase`
- `FeedbackEvent`
- `EvaluationRun`

## Hierarquia mínima de partes

```text
doll
├── head
│   ├── face
│   ├── hair
│   ├── ear_L
│   └── ear_R
├── torso
├── pelvis
├── arm_L
│   ├── upper_arm_L
│   ├── forearm_L
│   └── hand_L
├── arm_R
│   ├── upper_arm_R
│   ├── forearm_R
│   └── hand_R
├── leg_L
│   ├── thigh_L
│   ├── shin_L
│   └── foot_L
├── leg_R
│   ├── thigh_R
│   ├── shin_R
│   └── foot_R
└── clothing
    ├── top
    ├── bottom
    ├── footwear_L
    ├── footwear_R
    └── accessory
```

Nem todo projeto precisa usar todos os níveis.

## Side semantics

Valores:
- `left`
- `right`
- `center`
- `bilateral`
- `unknown`

Esquerda/direita são da perspectiva do personagem.

## Joint taxonomy inicial

- `fixed`
- `peg_socket`
- `ball_socket`
- `hinge`
- `swivel`
- `double_joint`
- `snap_fit`
- `friction_fit`
- `magnetic`
- `custom`

## Connector semantics

Um `JointInstance` representa comportamento cinemático.
Um `ConnectorInstance` representa geometria física do encaixe.

Exemplo:

```text
shoulder_L
  joint_type = ball_socket
  parent = torso
  child = upper_arm_L

connector pair:
  torso.shoulder_socket_L
  upper_arm_L.shoulder_ball
```

## Hidden geometry

Partes invisíveis em foto podem existir se forem necessárias à fabricação.

Elas DEVEM declarar provenance.

Exemplo:

```json
{
  "connector_id": "neck_socket_01",
  "provenance": {
    "type": "rule_based",
    "rule_id": "neck_ball_socket_v3"
  }
}
```

## Style families

O sistema deve suportar famílias de design:
- proporções corporais;
- tamanho relativo da cabeça;
- arredondamento;
- espessuras;
- linguagem de mãos/pés;
- roupas;
- joints preferidos.

Nunca codificar “PIPO” diretamente em funções genéricas.
Criar um `StyleRuleSet`.
