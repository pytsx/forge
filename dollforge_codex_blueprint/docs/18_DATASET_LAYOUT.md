# 18 — Dataset Layout

## Princípio

Separar dado bruto, proposta de máquina, correção humana e gold labels.

```text
datasets/
├── raw/
├── derived/
├── reviewed/
├── gold/
├── benchmarks/
└── manifests/
```

## Unidade de split

Nunca dividir aleatoriamente por imagem quando múltiplas imagens representam o mesmo boneco.

O split mínimo é por `character_instance_id` ou, preferencialmente, por projeto/família quando necessário para evitar leakage.

## Manifest

Cada dataset version deve registrar:
- lista de project_ids;
- artifact versions;
- ontology version;
- filters;
- exclusions;
- dedup strategy;
- generated_at;
- parent dataset;
- purpose.

## Dados úteis por projeto

- imagens originais;
- máscaras machine;
- máscaras corrigidas;
- observations;
- matches;
- DollGraph;
- meshes candidates;
- mesh aprovada;
- joints;
- connector params;
- manufacturing reports;
- feedback;
- fotos do protótipo físico, quando existirem.

## Dados sintéticos

Permitidos, mas sempre rotulados como synthetic.

Nunca misturar sintético e real sem conseguir medir cada distribuição separadamente.
