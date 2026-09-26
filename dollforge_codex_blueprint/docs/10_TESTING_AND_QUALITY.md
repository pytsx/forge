# 10 — Testes e Qualidade

## Pirâmide

### Unit
- schemas;
- transforms;
- nomenclatura;
- regras;
- geometry helpers.

### Contract
Cada adapter é testado contra sua interface.

### Integration
- segmentation -> matching;
- graph -> reconstruction;
- reconstruction -> Blender;
- Blender -> validator.

### Golden cases
Pequeno conjunto de projetos com outputs aprovados.

Comparar:
- labels;
- graph;
- transforms;
- medidas;
- joints;
- relatório.

### Physical tests
Quando houver impressão:
- medir encaixe;
- medir folga;
- força de inserção;
- range;
- quebra.

## Determinismo

Em testes:
- seed fixa;
- versões fixas;
- parâmetros fixos.

Para modelos não determinísticos:
- tolerâncias explícitas.

## Regressão

Toda correção de bug deve adicionar caso de regressão.
