# Skill: Ingestion Specialist

## Responsabilidade
Transformar arquivos de entrada em `ImageView` confiáveis.

## Deve
- preservar original;
- gerar hash;
- extrair metadata;
- normalizar orientação sem destruir original;
- avaliar blur/resolução;
- classificar qualidade;
- produzir thumbnails e masks preliminares quando configurado.

## Não deve
- segmentar semanticamente peças;
- inventar escala;
- modificar dataset raw.

## Output
`ImageView[]` + `ImageQAReport`.
