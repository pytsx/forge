# Perception Graph v1

A reconstrução não deve saltar diretamente de máscaras 2D para uma malha 3D.

Esta etapa cria uma descrição explícita e rastreável do objeto antes da volumetria.

## Posição no pipeline

```text
Segmentação
  ↓
Matching multi-view
  ↓
DollGraph
  ↓
PERCEPTION GRAPH (S09)
  ↓
Volumetria — próxima etapa
  ↓
Reconstrução
```

A reconstrução existente continua funcionando como baseline; o Perception Graph nasce como
um artefato independente para que a futura volumetria consuma uma descrição mais rica.

## O que o Perception Graph registra

### Objeto
- tipo: `stylized_modular_doll`;
- construção multi-part;
- estilo informado no projeto;
- simetria;
- regiões principais presentes.

### Peças
Para cada `part_instance_id`:
- classe e lado;
- região semântica;
- pai estrutural;
- observações correspondentes;
- medidas por vista;
- geometria resumida multi-view;
- família de forma;
- confiança e provenance.

### Evidência geométrica por vista
As máscaras aprováveis geram:
- centro normalizado;
- largura/altura relativas ao boneco naquela vista;
- área projetada;
- preenchimento da bbox;
- aderência da fronteira às bordas da imagem;
- perfil de largura em 16 cortes.

### Geometria multi-view
Frente/costas fornecem largura frontal. Esquerda/direita fornecem uma primeira aproximação
de profundidade projetada.

**Importante:** `depth_norm` ainda não é um depth map e não deve ser tratado como uma
superfície 3D. É evidência intermediária para a etapa de volumetria.

## Observado x inferido

Todo dado recebe `evidence_kind`:

- `observed`: medido diretamente em uma vista;
- `observed_multiview`: combinação de vistas;
- `human_confirmed`: derivado somente de observações aprovadas/corrigidas;
- `inferred`: relação inferida;
- `prior`: conhecimento estrutural/regra de bonecos.

Assim, uma silhueta lateral pode ser observada enquanto um encaixe de pescoço continua
explicitamente marcado como hipótese.

## Relações

O primeiro baseline produz relações como:
- `above`;
- `connected_to`;
- `attached_to`;
- `symmetric_to`.

Relações de posição usam as máscaras. Relações estruturais vindas do DollGraph mantêm o
status de inferência/prior correspondente.

## Interfaces

Cada joint do DollGraph gera uma `InterfaceHypothesis`, contendo:
- peças envolvidas;
- papel da interface;
- tipo de joint candidato;
- vistas nas quais as duas peças possuem observação;
- confiança;
- origem da hipótese.

A hipótese não significa que a geometria mecânica foi realmente observada.

## Próxima evolução

A PR de volumetria deve consumir este artefato para combinar:
- silhuetas;
- câmeras;
- perfis de forma;
- depth estimation;
- surface normals;
- visual hull / occupancy / SDF.

Posteriormente, um adapter multimodal poderá enriquecer o mesmo contrato com leitura
semântica mais sofisticada sem substituir as evidências geométricas verificáveis.
