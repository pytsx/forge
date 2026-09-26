# 09 — Infraestrutura Local

## Objetivo

Operar localmente, com serviços separados.

## Serviços sugeridos

- `api`: FastAPI
- `orchestrator`: Prefect ou Dagster
- `metadata_db`: PostgreSQL
- `artifact_store`: filesystem inicialmente; MinIO depois
- `vector_store`: pgvector ou FAISS
- `ml_segmentation`
- `ml_matching`
- `ml_reconstruction`
- `blender_worker`
- `review_ui`
- `training_worker`
- `mlflow`: experiment tracking opcional/recomendado

## Ambiente

Separar ambientes:
- app/orchestrator;
- CV models;
- reconstruction models;
- Blender Python.

Isso reduz conflito de CUDA/PyTorch/Python.

## GPU

Criar abstraction `ComputeProfile`:
- cpu
- cuda_low_vram
- cuda_standard
- cuda_high_vram

Model adapters informam requisito.

## Artifact Store

Estrutura conceitual:

```text
projects/<project_id>/
  raw/
  normalized/
  masks/
  correspondences/
  graph/
  reconstructions/
  connectors/
  assemblies/
  blender/
  exports/
  reports/
  feedback/
```

Nunca sobrescrever artefatos; criar novas versões.
