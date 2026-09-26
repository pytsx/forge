# 13 — Biblioteca de Referências

## Tipos

### VisualReference
- imagens por ângulo;
- máscaras;
- landmarks;
- estilo;
- iluminação metadata.

### GeometryReference
- mesh;
- medidas;
- topology info;
- canonical transform.

### JointReference
- joint type;
- connector pair;
- dimensões;
- tolerância;
- material/processo.

### CaseReference
- projeto completo aprovado;
- inputs;
- outputs;
- feedback;
- decisões.

## Retrieval

Consulta típica para reconstruir mão:

```text
class = hand
side = left
style_family = pipo_v1
size_bucket = small
similarity = embedding(image crop)
manufacturing_profile = resin_v1
```

## Gold status

Referência só pode ser `gold` quando:
- aprovada;
- provenance completo;
- sem erro crítico conhecido;
- versão de ontologia compatível.

## Duplicatas

Deduplicar visualmente e por origem para evitar overweight de um mesmo boneco.
