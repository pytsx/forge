# 03 — Contratos de Dados

## Filosofia

A estabilidade do projeto deve vir dos contratos e não de uma implementação específica.

## Artefatos fundamentais

### ImageView
Representa uma imagem original e seus metadados.

### PartObservation
Representa uma parte percebida em UMA vista.

### PartInstance
Representa a entidade física compartilhada entre múltiplas vistas.

Exemplo:

```text
front.mask_12
left.mask_07
back.mask_14
       |
       v
part_instance = hand_L
```

### DollGraph

Grafo estrutural final antes da geração de geometria.

Nós:
- peças;
- joints;
- connectors.

Arestas:
- parent/child;
- connection;
- symmetry;
- correspondence.

## Nunca confundir

`PartObservation != PartInstance`

Uma observation pode estar errada.
Uma instance é a hipótese consolidada.

## Versionamento

Schemas seguem SemVer.

Breaking changes:
- removem campos;
- alteram semântica;
- mudam sistema de coordenadas;
- mudam unidade.

## Lineage

Todo output deve conter:
- input artifact ids;
- código/commit;
- model version;
- parameters hash;
- seed;
- timestamp;
- environment fingerprint.
