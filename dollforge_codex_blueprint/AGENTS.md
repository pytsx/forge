# AGENTS.md — Regras obrigatórias para Codex

Este arquivo é a autoridade operacional do repositório.

## Missão

Construir um sistema local, modular e especializado em reconstrução, engenharia e fabricação de bonecos articulados a partir de múltiplas imagens.

## O Codex deve

- preservar modularidade entre stages;
- implementar interfaces antes de otimizar modelos;
- validar todo payload com schemas;
- manter datasets brutos imutáveis;
- separar inference, review, training e production;
- registrar provenance em toda inferência;
- criar testes para cada contrato;
- preferir componentes substituíveis via adapters;
- manter Blender desacoplado do runtime de ML;
- produzir logs estruturados;
- versionar artefatos e parâmetros;
- deixar comportamento determinístico quando possível;
- documentar mudanças de arquitetura em ADRs.

## O Codex não deve

- criar um modelo monolítico end-to-end como primeira solução;
- acoplar lógica de negócio diretamente a um modelo específico;
- sobrescrever imagens, máscaras ou meshes originais;
- promover feedback não revisado diretamente para dataset de treino;
- executar fine-tuning automaticamente em produção;
- inferir encaixes ocultos sem registrar confiança e origem;
- usar apenas uma nota final como sinal de qualidade;
- misturar coordenadas ou unidades sem conversão explícita;
- assumir que esquerda/direita é do observador;
- corrigir manualmente uma mesh sem gerar um delta/registro da correção;
- aceitar mesh inválida como sucesso só porque “parece boa”.

## Convenções globais

- Unidade canônica: milímetros.
- Sistema canônico 3D: Z-up, right-handed.
- Left/Right: sempre da perspectiva do personagem.
- Origem preferencial: centro do pelvis/root.
- Forward do personagem: -Y no espaço canônico do projeto.
- Identificadores: UUID ou ULID persistente; nomes humanos nunca são IDs.
- Datas: ISO-8601 UTC.
- Scores: intervalo [0, 1], salvo quando schema declarar outro formato.
- Todo artefato deve carregar `project_id`, `run_id`, `stage`, `version`.

## Provenance obrigatório

Qualquer dado inferido deve declarar um dos tipos:

- `observed`: diretamente suportado pelas imagens;
- `derived_geometry`: calculado geometricamente;
- `retrieved_reference`: recuperado da biblioteca;
- `rule_based`: produzido por regra;
- `model_inferred`: inferido por modelo;
- `human_edited`: alterado por humano;
- `human_approved`: explicitamente aprovado.

## Confiança

Decisões automáticas críticas devem registrar `confidence`.

Baixa confiança deve:
1. gerar estado `needs_review`;
2. impedir promoção automática;
3. permitir alternativas candidatas.

## Política de mudanças

Mudanças em:
- ontologia;
- schema;
- sistema de coordenadas;
- regras de fabricação;
- semântica dos labels;

exigem migration ou versão nova.

## Definição de pronto

Uma feature só está pronta quando contém:
- código;
- tipos;
- validação;
- testes;
- logs;
- documentação;
- exemplo mínimo reproduzível.
