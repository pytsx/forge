# Skill: Cross-view Correspondence Specialist

## Responsabilidade
Dar match entre `PartObservation` de diferentes vistas.

## Sinais
- semantic class;
- side;
- embeddings;
- relative body location;
- camera geometry;
- colors/material;
- contour;
- symmetry.

## Output
`PartInstance[]` + correspondences.

## Failure mode
Quando ambíguo:
- retornar múltiplos candidatos;
- não forçar match;
- solicitar review.
