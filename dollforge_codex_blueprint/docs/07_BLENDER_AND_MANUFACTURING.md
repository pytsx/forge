# 07 — Blender e Fabricação

## Papel do Blender

Blender é o runtime procedural para:
- assembly;
- modifiers;
- booleans;
- geração de sockets;
- remesh;
- smoothing;
- inspeção;
- export;
- arquivo final editável.

Não executar modelos pesados dentro do processo principal do Blender.

## Integração preferida

```text
orchestrator
   |
   +--> ML workers
   |
   +--> artifact store
   |
   +--> blender headless worker
             |
             +--> .blend
             +--> .stl/.3mf/.glb
             +--> validation report
```

## Collections sugeridas

```text
DOLL_ROOT
├── 00_REFERENCES
├── 10_BODY
├── 20_CLOTHING
├── 30_CONNECTORS
├── 40_GUIDES
├── 50_BOOLEAN_CUTTERS
├── 60_PRINT_EXPORTS
└── 90_DEBUG
```

## Nomes

`<part_class>__<side>__<instance_short_id>`

Exemplo:
`upper_arm__left__01HXY9`

## Geometria paramétrica

Joints devem ser gerados por parâmetros:
- diameter;
- length;
- insertion_depth;
- clearance;
- chamfer;
- fillet;
- retention;
- rotation_axis.

Nunca “esculpir” um connector recorrente sem parametrização.

## Tolerância

Clearance não é constante universal.

Depende de:
- material;
- processo;
- orientação;
- acabamento;
- escala;
- função do joint.

O projeto deve manter perfis de fabricação.

## Validação

Antes de status `manufacturable`, verificar:
- closed/manifold;
- normals;
- zero-area faces;
- self-intersections relevantes;
- mínimo de parede;
- connector fit;
- colisão em pose neutra;
- range de movimento;
- separabilidade física.
