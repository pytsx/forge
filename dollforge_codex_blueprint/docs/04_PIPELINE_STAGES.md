# 04 — Stages do Pipeline

## S00 — Project Intake
Entrada:
- imagens;
- dimensões conhecidas opcionais;
- style family;
- objetivo de fabricação.

Saída:
- `DollProject`.

## S01 — Image QA
Detecta:
- blur;
- recorte;
- resolução;
- oclusão;
- fundo problemático;
- inconsistência extrema.

Não rejeitar automaticamente sem registrar motivo.

## S02 — Foreground Segmentation
Cria máscara do boneco completo.

## S03 — Camera / View Understanding
Classifica:
- front;
- back;
- left;
- right;
- 3/4;
- detail;
- unknown.

Estima parâmetros de câmera quando possível.

## S04 — Semantic Part Detection
Propõe boxes/labels.

## S05 — Fine Part Segmentation
Gera máscaras por peça.

## S06 — Cross-view Matching
Agrupa observations em `PartInstance`.

Combinar:
- classe semântica;
- side;
- embedding visual;
- posição relativa;
- aparência;
- geometria epipolar/câmera;
- simetria;
- regras do DollGraph.

## S07 — Scale Alignment
Usar:
1. marcador físico conhecido;
2. dimensão fornecida;
3. calibração de câmera;
4. proporções canônicas como fallback.

Nunca fingir escala absoluta quando só há escala relativa.

## S08 — DollGraph Synthesis
Consolida:
- parts;
- symmetry;
- joints prováveis;
- hierarchy.

Checkpoint humano recomendado.

## S09 — Reference Retrieval
Busca casos aprovados similares por:
- classe da peça;
- shape embedding;
- style family;
- tamanho;
- tipo de joint;
- fabricação.

## S10 — Per-part 3D Reconstruction
Cada part class usa adapter próprio.

Deve aceitar múltiplas vistas recortadas e opcionalmente um template.

## S11 — Shape Regularization
Aplicar:
- simetria;
- smoothness;
- template constraints;
- espessura;
- simplificação.

## S12 — Joint Inference
Produz joint candidates com:
- tipo;
- eixo;
- range;
- confiança;
- provenance.

## S13 — Connector Generation
Gera geometria paramétrica para joints.

## S14 — Assembly Optimization
Resolve:
- posições;
- gaps;
- colisões;
- interpenetração;
- alinhamento.

## S15 — Blender Build
Gera collections e objetos nomeados.

## S16 — Manufacturing Validation
Checa:
- manifold;
- normals;
- wall thickness;
- minimum feature size;
- clearances;
- collisions;
- unsupported connector geometry;
- tolerâncias.

## S17 — Human Final Review
Avaliação multi-dimensional.

## S18 — Knowledge Promotion
Apenas artefatos aprovados podem virar referência “gold”.

## S19 — Offline Learning
Treinamento versionado, nunca implícito.
