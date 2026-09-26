# Calibrated Multi-view Volumetry v2

Esta etapa substitui proporções 2D independentes por um sistema multi-view canônico e
fisicamente coerente.

## Pipeline

```text
Máscaras multi-view
  ↓
S07C Calibração por vista
  ↓
Sistema de coordenadas canônico
  ↓
S09V Visual Hull voxelado
  ↓
Occupancy Grid
  ↓
Signed Distance Field
  ↓
Marching Cubes / superfície zero
  ↓
Reprojeção nas referências
  ↓
Métricas de fidelidade
  ↓
Reconstrução
```

## Calibração

Para cada vista ortográfica:

```text
s_v = H_canonical / H_view_px
```

Cada câmera guarda:

- `world_units_per_pixel`;
- `mm_per_pixel` quando a escala é física;
- principal point;
- transformação `world_from_view`;
- convenção de yaw;
- confiança.

Convenção:

```text
FRONT   0°
RIGHT  90°
BACK  180°
LEFT  -90°
```

O pixel horizontal de FRONT/BACK representa X.
O pixel horizontal de LEFT/RIGHT representa Y.
O pixel vertical representa Z.

Toda projeção 3D ↔ 2D está centralizada em
`volumetry/projection.py`.

## Visual Hull real

A v1 aproximava cada peça por perfis suavizados. A v2 cria um grid voxelado e testa
explicitamente a compatibilidade com as silhuetas.

```text
occupancy(x) = compatível com as evidências visuais calibradas
```

Máscaras aprovadas/corrigidas possuem poder de carving forte. Evidências automáticas
entram com peso de confiança. Priors e relações inferidas não removem voxels porque não
participam do carving.

## SDF

O occupancy vira um campo assinado:

```text
inside  = distance_transform_edt(occupancy)
outside = distance_transform_edt(~occupancy)
sdf     = (outside - inside) * voxel_size
```

Convenção:

- negativo dentro;
- zero na superfície;
- positivo fora.

A superfície final é extraída do nível zero do SDF.

## Persistência

O campo não é incluído no JSON do `VolumeCandidate`.

Para cada peça é salvo um artefato `.npz` contendo:

- `occupancy`;
- `sdf`;
- `origin_xyz`;
- `voxel_size`;
- `shape_xyz`.

O JSON contém somente um `VolumeFieldDescriptor` apontando para esse artefato.

## Reprojeção

Cada occupancy é projetado de volta nas vistas utilizadas.

Métricas por vista:

- silhouette IoU;
- erro relativo de área;
- boundary RMSE em pixels.

A confidence da volumetria passa a combinar:

```text
30% percepção
20% câmera
20% cobertura
30% reprojeção
```

Esses pesos são um baseline explícito, não uma regra definitiva.

## Validação

A validação agora separa duas perguntas:

```text
MESH VALID
!=
GEOMETRY ACCURATE
```

Além de manifold/normais/faces, o relatório inclui:

- calibração multi-view;
- reprojeção por vista;
- consistência multi-view média.

## Resolução

A configuração aceita 32³ a 128³:

- preview: 32;
- padrão: 64;
- alta qualidade: 128.

O default desta versão é 64.

## Limite

`concavity_support = false` continua correto.

O Visual Hull não recupera cavidades que não afetam a silhueta. A etapa seguinte deve
refinar o SDF com depth maps, surface normals e termos de regularização sem perder a
restrição imposta pelas referências.
