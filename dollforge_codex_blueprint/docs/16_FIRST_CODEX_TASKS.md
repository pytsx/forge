# 16 — Primeiras tarefas para o Codex

## PR 1 — Foundation
Criar:
- package Python;
- Pydantic models;
- enums;
- IDs;
- schemas;
- tests.

Sem ML.

## PR 2 — Artifact Store
Criar storage versionado local:
- put;
- get;
- metadata;
- hash;
- lineage.

## PR 3 — Project Intake
CLI/API:
`dollforge project create`
`dollforge view add`

## PR 4 — Stage Engine
Executar DAG simples e persistir status.

## PR 5 — Segmentation Adapter Mock
Antes de modelo real:
- adapter fake;
- masks fixture;
- review workflow.

## PR 6 — SAM/Grounding Adapter
Implementar sem acoplar aos contratos.

## PR 7 — Correspondence Baseline
Regras semânticas + embeddings.

## PR 8 — DollGraph
Builder + review.

## PR 9 — Blender Worker
Headless:
- abrir;
- importar;
- criar collections;
- salvar.

## PR 10 — Reconstruction Baseline
Adicionar um modelo image-to-3D por adapter.

A disciplina aqui é intencional: primeiro construir a fábrica, depois colocar máquinas melhores nela.
