# DollForge — Codex Engineering Blueprint

Este pacote define a arquitetura, regras, contratos, skills e plano de implementação para um sistema local e modular de reconstrução 3D especializado em bonecos articulados.

## Objetivo

Transformar múltiplas imagens de um mesmo boneco (frente, laterais, costas, 3/4 e detalhes) em um conjunto de peças 3D coerentes, montáveis, fabricáveis e consistentes com as regras do universo de produto.

O sistema NÃO deve ser tratado como um simples pipeline `imagem -> mesh`.

A abstração correta é:

`imagens -> observações -> peças semânticas -> correspondência multi-view -> geometria por peça -> juntas/encaixes -> montagem -> validação -> fabricação -> feedback -> aprendizado`

## Princípios inegociáveis

1. Modularidade: cada etapa é isolável, testável e substituível.
2. Especialização: cada modelo resolve uma função específica.
3. Contratos fortes: módulos trocam dados por schemas versionados.
4. Humano no loop: cada etapa crítica pode ser revisada e reprocessada.
5. Rastreabilidade: toda decisão deve ter provenance.
6. Fabricação primeiro: uma mesh visualmente bonita mas não fabricável é falha.
7. Conhecimento explícito: juntas, folgas, simetria, proporções e regras não ficam “escondidas” em prompts.
8. Aprendizado seguro: dados não aprovados não entram automaticamente em treino.
9. Reprodutibilidade: seeds, versões, parâmetros, pesos e artefatos precisam ser registrados.
10. O Blender é a camada procedural/final de modelagem, validação e assembly; não o orquestrador de IA.

## Ordem recomendada para o Codex

Leia nesta ordem:

1. `AGENTS.md`
2. `docs/00_PRODUCT_VISION.md`
3. `docs/01_SYSTEM_ARCHITECTURE.md`
4. `docs/02_DOLL_ONTOLOGY.md`
5. `docs/03_DATA_CONTRACTS.md`
6. `docs/04_PIPELINE_STAGES.md`
7. `docs/05_HUMAN_IN_THE_LOOP.md`
8. `docs/06_KNOWLEDGE_AND_LEARNING.md`
9. `docs/07_BLENDER_AND_MANUFACTURING.md`
10. `docs/08_EVALUATION.md`
11. `docs/09_LOCAL_INFRA.md`
12. `docs/10_TESTING_AND_QUALITY.md`
13. `docs/11_ROADMAP.md`
14. `docs/12_MODEL_CANDIDATES.md`
15. `rules/*.md`
16. `skills/*/SKILL.md`

## Primeira entrega esperada

O primeiro objetivo NÃO é gerar um boneco perfeito.

O primeiro objetivo é produzir um pipeline ponta a ponta que:

- ingere 4 vistas;
- registra câmeras e metadados;
- segmenta as principais peças;
- permite correção humana;
- cria `part_instance_id` consistentes entre as vistas;
- gera um `DollGraph`;
- exporta um projeto Blender com objetos nomeados e organizados;
- registra feedback e lineage.

Somente depois disso deve começar a reconstrução geométrica avançada.

## Nome do projeto

Nome de trabalho: `DollForge`.

Pode ser alterado sem afetar os contratos.
