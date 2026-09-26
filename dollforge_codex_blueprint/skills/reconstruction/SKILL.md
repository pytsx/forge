# Skill: Per-Part 3D Reconstruction Specialist

## Responsabilidade
Reconstruir UMA `PartInstance`.

## Input
- crops multi-view;
- masks;
- cameras;
- reference candidates;
- part class;
- constraints.

## Output
- mesh candidates;
- confidence;
- reprojection metrics;
- provenance.

## Política
Permitir adapters específicos:
- HeadReconstructor
- TorsoReconstructor
- LimbReconstructor
- HandReconstructor
- FootwearReconstructor
- ClothingReconstructor

Não tentar impor um único modelo a todas as classes.
