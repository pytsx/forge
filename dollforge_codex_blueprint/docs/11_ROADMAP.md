# 11 — Roadmap de Implementação

## Milestone 0 — Repository Foundation
- package structure;
- schemas;
- IDs;
- artifact store;
- logging;
- test harness;
- CI local.

## Milestone 1 — Intake + Reviewable Segmentation
- ingest multi-view;
- image QA;
- view labels;
- whole-object mask;
- part proposal;
- human mask correction;
- save feedback.

Definition of done:
usuário consegue corrigir partes e persistir alterações.

## Milestone 2 — Cross-view Identity
- part instances;
- matching candidates;
- confidence;
- human rematch;
- DollGraph v1.

## Milestone 3 — Scale + Camera
- calibration;
- relative scale;
- optional known dimension;
- consistent canonical transforms.

## Milestone 4 — Reconstruction Baseline
- one adapter generic;
- part crops;
- multi-view reconstruction;
- import meshes;
- visualize in Blender.

## Milestone 5 — Specialist Reconstruction
Separar:
- head;
- torso;
- limbs;
- hands;
- feet/shoes;
- clothing.

## Milestone 6 — Joint Knowledge
- joint ontology;
- retrieval;
- rule engine;
- connector library.

## Milestone 7 — Assembly
- canonical skeleton;
- fit;
- symmetry;
- collision minimization;
- neutral pose.

## Milestone 8 — Manufacturing
- validation;
- tolerances;
- exports;
- physical feedback.

## Milestone 9 — Learning
- gold dataset;
- evaluation registry;
- fine-tuning;
- model promotion gates.

## Milestone 10 — Single View
Somente iniciar após benchmark multi-view estável.
