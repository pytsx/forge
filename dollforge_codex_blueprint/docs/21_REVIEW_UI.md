# 21 — Review UI

## Objetivo

Fazer o humano corrigir o sistema com o menor esforço possível e gerar labels de alta qualidade.

## Tela de segmentação

Mostrar:
- imagem;
- masks;
- labels;
- confidence;
- toggle de peças;
- ferramentas brush/add/remove;
- history.

## Tela multi-view

Mostrar vistas sincronizadas.
Ao selecionar uma `PartInstance`, destacar todas as observations correspondentes.

Permitir:
- match;
- unmatch;
- swap left/right;
- merge/split.

## Tela 3D

Mostrar:
- source silhouettes;
- candidate meshes;
- reprojection;
- referência recuperada;
- sliders de parâmetros.

## Tela de joints

Mostrar:
- joint axis;
- range;
- connector dimensions;
- cutaway;
- source of inference.

## Feedback UX

Evitar depender de texto.
Usar reason codes + parâmetros + texto opcional.

## Comparação

Quando reprocessado:
- A/B;
- diff de parâmetros;
- diff de métricas;
- manter histórico.
