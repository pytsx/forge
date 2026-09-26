# Volumetry Stage v1

A etapa S09V transforma as descrições do Perception Graph e as máscaras multi-view em um
volume contínuo antes da reconstrução final.

## Fluxo

```text
Perception Graph
      ↓
máscaras da mesma peça em várias vistas
      ↓
amostragem de perfis por altura
      ↓
fusão frente/costas + esquerda/direita
      ↓
seções transversais
      ↓
visual hull suavizado
      ↓
silhouette_volume_mesh_v1
      ↓
Modelo 3D
```

## O que muda em relação aos templates

O reconstrutor anterior escolhia uma forma semântica fixa — por exemplo um
superelipsoide de cabeça — e apenas ajustava suas dimensões.

O novo fluxo mede a própria máscara em várias alturas. Para cada seção registra:

- deslocamento horizontal frontal;
- meia largura;
- deslocamento em profundidade;
- meia profundidade;
- confiança.

Assim duas cabeças com a mesma bounding box, mas contornos diferentes, produzem
superfícies diferentes.

## Vistas opostas

- frente e costas são alinhadas no eixo X;
- esquerda e direita são alinhadas no eixo Y;
- vistas opostas contribuem por mediana, reduzindo a influência de uma única máscara;
- correções humanas propagadas para outras vistas entram automaticamente na próxima execução.

## Centro e escala

O centro mundial da peça é estimado a partir da posição da máscara dentro da silhueta
completa do personagem em cada vista. A escala usa a altura canônica do DollGraph:

- em projetos sem medida física, a unidade continua relativa;
- com altura fornecida, as dimensões do volume são produzidas em milímetros.

## Limitação deliberada

Visual hull recupera muito bem aquilo que altera a silhueta, mas **não pode recuperar uma
cavidade totalmente interna ao contorno**.

Por isso cada volume possui:

`concavity_support = false`

Nesta versão isso é uma informação explícita, não uma lacuna escondida. A próxima camada
de volumetria deve acrescentar:

- depth maps;
- surface normals;
- cues de iluminação/curvatura;
- fusão depth + silhouette;
- correção humana de relevos e cavidades.

## UI

A aba **Volumetria** mostra para cada peça:

- dimensões X × Y × Z;
- vistas utilizadas;
- número de seções;
- confiança;
- gráfico de largura e profundidade ao longo da altura;
- status de suporte a concavidades.

## Adapters

- `silhouette_visual_hull_v1`: fusão volumétrica;
- `silhouette_volume_mesh_v1`: promove a superfície do volume para a reconstrução.

O antigo `doll_templates_multiview_v2` continua disponível como fallback.
