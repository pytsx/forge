# 01 — Arquitetura do Sistema

## Macroarquitetura

```text
Images
  |
  v
Ingestion / Normalization
  |
  v
Camera & Scale Estimation
  |
  v
Part Detection + Segmentation
  |
  v
Cross-view Correspondence
  |
  v
DollGraph Builder
  |
  +----------------------+
  |                      |
  v                      v
Reference Retrieval     Human Review
  |                      |
  +----------+-----------+
             |
             v
Per-part Reconstruction
             |
             v
Joint / Connector Synthesis
             |
             v
Assembly Optimization
             |
             v
Blender Procedural Build
             |
             v
Manufacturing Validation
             |
             v
Human Evaluation
             |
             v
Validated Knowledge Store
             |
             +--> Offline training / evaluation
```

## Separação de processos

### ML Runtime
Responsável por:
- percepção;
- embeddings;
- matching;
- reconstrução;
- inferência.

### Orchestrator
Responsável por:
- DAG;
- retries;
- cache;
- checkpoints;
- estado;
- lineage.

### Knowledge Service
Responsável por:
- referências visuais;
- peças 3D;
- parâmetros;
- casos aprovados;
- retrieval.

### Review UI
Responsável por:
- revisão humana;
- correções;
- comparações;
- notas;
- aprovação.

### Blender Worker
Responsável por:
- importar meshes;
- normalizar;
- aplicar regras procedurais;
- gerar connectors;
- organizar collections;
- medir colisões;
- salvar `.blend`;
- exportar formatos de fabricação.

## Regra de isolamento

Cada stage recebe um input versionado e devolve um output versionado.

Nenhum stage deve acessar tabelas internas de outro stage para “facilitar”.
Toda dependência passa por contrato explícito.
