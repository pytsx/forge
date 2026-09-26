# 06 — Conhecimento e Aprendizado

## Dois loops distintos

### Loop rápido — Knowledge Retrieval
Acontece imediatamente.

Quando um projeto é aprovado:
- indexar referências visuais;
- indexar meshes;
- registrar parâmetros;
- registrar joints;
- registrar correções;
- criar embeddings.

Isso melhora os próximos projetos sem mudar pesos.

### Loop lento — Model Training
Acontece offline, controlado.

Fluxo:
1. selecionar dados aprovados;
2. congelar dataset version;
3. treinar candidato;
4. avaliar em benchmark fixo;
5. comparar com produção;
6. revisão humana;
7. promover ou rejeitar.

## Estados dos dados

- `raw`
- `machine_proposed`
- `human_corrected`
- `human_approved`
- `gold`
- `deprecated`

Somente `gold` e conjuntos explicitamente autorizados entram por padrão em treinamento.

## Active learning

Priorizar revisão humana de:
- baixa confiança;
- desacordo entre modelos;
- casos fora da distribuição;
- joints inéditos;
- maiores perdas de reconstrução;
- peças com alta taxa de retrabalho.

## Evitar feedback loops ruins

Não treinar diretamente com:
- outputs automáticos não revisados;
- notas sem contexto;
- dados duplicados;
- projetos com erro de escala conhecido;
- versões antigas de ontologia sem migração.

## Case-based reasoning

Antes de gerar connector oculto:
1. recuperar casos parecidos;
2. avaliar compatibilidade de escala;
3. avaliar style family;
4. aplicar regra;
5. registrar referências usadas.

## Preference learning futuro

Pode-se aprender preferências humanas para:
- arredondamento;
- proporção;
- estilização;
- simplificação.

Mas regras de segurança mecânica/fabricação continuam como constraints explícitas.
