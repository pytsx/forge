# 08 — Avaliação

## Métricas por stage

### Segmentação
- IoU por classe;
- boundary F-score;
- taxa de correção humana;
- tempo de correção.

### Cross-view
- match precision;
- match recall;
- identity switches;
- taxa de relabel.

### Reconstrução
Quando ground truth existir:
- Chamfer distance;
- normal consistency;
- silhouette error;
- keypoint error.

Sem ground truth:
- reprojection consistency;
- multi-view silhouette agreement;
- symmetry residual;
- human score.

### Joints
- classification accuracy;
- axis error;
- center error;
- range error;
- human override rate.

### Fabricação
- manifold pass;
- wall thickness pass;
- clearance pass;
- assembly collision pass;
- print success;
- physical assembly success.

## Benchmark

Criar conjunto fixo e versionado com:
- bonecos fáceis;
- bonecos difíceis;
- roupas;
- oclusões;
- estilos variados;
- joints visíveis e ocultos.

Não usar benchmark como treino.

## Gate de produção

Um modelo novo só substitui o atual se:
- não regredir métricas críticas;
- melhorar objetivo declarado;
- passar teste físico ou proxy definido;
- passar revisão humana amostrada.
