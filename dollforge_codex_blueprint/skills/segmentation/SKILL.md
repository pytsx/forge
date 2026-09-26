# Skill: Doll Part Segmentation Specialist

## Responsabilidade
Detectar e segmentar partes semânticas de bonecos.

## Estratégia
1. grounding/detection;
2. segmentation refinada;
3. constraints de ontologia;
4. confidence;
5. revisão humana.

## Inputs
- `ImageView`
- ontology version
- style hints opcionais

## Outputs
- `PartObservation[]`

## Regras
- não consolidar IDs entre vistas;
- side pode ser unknown;
- máscara original do modelo deve ser preservada após correção humana;
- guardar logits/scores quando úteis para aprendizado.
