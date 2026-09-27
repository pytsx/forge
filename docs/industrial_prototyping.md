# Impressão 3D e prototipagem: análise e implementação

## Diagnóstico

O DollForge já tem uma base útil: contratos tipados, artefatos imutáveis com hashes,
histórico de correções, cache, reconstrução por peça e critérios de qualidade das
silhuetas. Entretanto, o produto continua especializado em personagens/bonecos.
A reconstrução de quatro imagens não equivale a um modelo CAD dimensionalmente
qualificado para qualquer peça industrial.

As principais lacunas encontradas foram:

1. A inspeção S16 verificava apenas fechamento, orientação e área das faces.
   Não detectava explicitamente peças faltantes, identidades duplicadas, limites
   de complexidade, componentes separados ou incompatibilidade com a impressora.
2. A S16 retornava sempre `needs_review`, mesmo diante de falhas. A interface
   agrupava verificações sem mostrar a peça e o motivo de cada problema.
3. Os limites de reprojeção da S16 estavam fixos e podiam divergir dos limites
   configurados para a volumetria S09V.
4. `Service.execute` mantinha o bloqueio global do armazenamento durante toda a
   reconstrução, impedindo consultas de progresso durante trabalho pesado.
5. A grade volumétrica alocava três matrizes completas de coordenadas, além da
   matriz final; a projeção criava temporários proporcionais à grade inteira.

## Implementado

### Critérios rastreáveis por execução

`PipelineConfig.print_profile` registra nome, processo, material, volume útil XYZ,
tamanho máximo de voxel, limite de faces por peça e exigência de um único
componente conectado. Processo/material são metadados: não ativam regras físicas
específicas de FDM, SLA ou SLS.

Os limites físicos são opcionais e não são inventados pelo sistema. Quando não
informados, as verificações correspondentes permanecem pendentes. O limite padrão
de 1.000.000 faces por peça é um orçamento computacional, não uma especificação
de impressora. Acima dele, a inspeção bloqueia a peça antes da análise topológica.

Na interface, **Executar pipeline** abre os critérios da execução. Uma nova
execução reaproveita a configuração da execução selecionada; replay mantém o
snapshot original. Projetos e manifests antigos continuam legíveis pelos defaults
dos novos campos. Novos campos de configuração/código invalidam o cache anterior.

Exemplo de corpo para `POST /api/projects/{id}/runs` (valores ilustrativos,
devem ser substituídos pelos limites do processo):

```json
{
  "volumetry_resolution": 96,
  "print_profile": {
    "name": "Protótipo FDM — bancada A",
    "process": "fdm",
    "material": "PLA",
    "build_volume_mm": [200, 200, 200],
    "max_voxel_size_mm": 0.3,
    "max_faces_per_part": 1000000,
    "require_single_shell": true
  }
}
```

### Inspeção e bloqueio

A inspeção não modifica nem repara a geometria. Verifica:

- existência, unicidade e correspondência com as peças esperadas no grafo;
- índices das faces, coordenadas finitas, faces duplicadas e degeneradas;
- fechamento por arestas, componentes conectados e orientação de cada componente;
- escala física, dimensões XYZ e compatibilidade com o volume útil na orientação atual;
- evidência do tamanho do voxel por peça quando há limite configurado;
- reprojeção com os limites da própria execução.

Dimensões, área, volume geométrico, contagem de faces e componentes ficam no relatório.
Volume é informado somente quando os critérios implementados de orientação e
fechamento são satisfeitos e não há faces duplicadas. Autointerseções ainda podem
invalidar esse valor: ele não deve ser usado como estimativa de consumo, massa
impressa ou certificado de volume. Dimensões ignoram vértices não usados por faces.

O contrato de reconstrução usa vértices já transformados em coordenadas canônicas;
a inspeção não aplica `transform` novamente. Dimensões não são calculadas como mm
quando a escala é relativa. O teste de envelope não busca orientações alternativas
e não inclui suportes, brim, raft ou afastamentos de máquina.

Com falha, `geometry_status=blocked`, a execução permanece em revisão e não gera
o projeto Blender. Malhas intermediárias e relatórios continuam exportáveis como
material de diagnóstico. Sem falhas detectadas, `geometry_status=passed` significa
apenas aprovação nos testes geométricos implementados; `manufacturable` permanece
`false`, com verificações físicas pendentes explicitadas.

O pacote ZIP contém `inspection.json` e `inspection.csv`, além de manifest,
artefatos e metadados. Downloads de execuções em andamento ou invalidadas são
bloqueados para evitar pacotes inconsistentes.

### Desempenho e responsividade

- Um bloqueio específico serializa as execuções. O armazenamento fica bloqueado
  apenas durante suas operações; a interface pode consultar progresso durante a
  reconstrução. Uma execução concluída não pode ser iniciada novamente pelo mesmo ID.
- A grade usa uma única matriz de coordenadas. A projeção de máscaras trabalha
  com blocos de até 65.536 pontos, preservando precisão `float64`, ordem da grade
  e regras de arredondamento.

Benchmark local em 26/09/2026, Python 3.11.9, NumPy 2.4.6, grade 128³,
três repetições, criação de grade + uma projeção:

| Implementação | Pico de alocações rastreadas | Tempo mediano |
| --- | ---: | ---: |
| Anterior, densa | 160,00 MiB | 0,2885 s |
| Buffers limitados | 55,57 MiB | 0,2315 s |

Redução de aproximadamente 65% no pico e 20% no tempo desse microbenchmark. As
saídas tiveram SHA-256 idêntico. Esses números não medem RSS total nem tempo do
pipeline completo; SDF, reprojeção de qualidade, serialização e Blender continuam
tendo custos próprios. A medição pode variar por máquina e carga.

Reproduzir no ambiente instalado do projeto:

```powershell
.venv/Scripts/python.exe scripts/benchmark_projection.py --resolution 128 --repeats 3
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts
node --check src/dollforge/web/app.js
```

Os testes cobrem defeitos geométricos, limites físicos, relatórios exportados,
consultas concorrentes, igualdade da projeção e integração do pipeline com o
bloqueio da geração Blender. O teste de integração usa peças sintéticas e um
adaptador Blender simulado; não qualifica peças impressas nem o Blender real.

## Próximos passos para adoção industrial

1. **Referência dimensional:** entrada de medidas por característica, escala
   verificada e comparação com referência CAD/escaneamento medido. Separar
   resolução numérica, erro da reconstrução e erro do processo de impressão.
2. **Preparação para impressão:** autointerseções, manifold de vértices,
   espessura local, canais/cavidades, folgas e colisões; depois orientação,
   suportes e integração com fatiador. Um único componente conectado e arestas
   fechadas não bastam para provar ausência desses defeitos.
3. **Qualificação por máquina/material:** peças de teste físicas, medidas e
   limites aprovados por aplicação. Registrar impressora, material/lote,
   parâmetros, pós-processamento e resultados de medição; versionar o perfil.
4. **Liberação formal:** aprovação técnica vinculada aos hashes das peças e
   das evidências, com estado separado da revisão estética. Exportação destinada
   à produção deve exigir essa liberação quando esse fluxo for implementado.
5. **Piloto com o público-alvo:** acompanhar tempo até o primeiro protótipo
   utilizável, retrabalho por peça, falhas de fatiamento, desvios medidos e taxa
   de aceitação. Ajustar o produto com projetistas, operadores e responsáveis
   pela qualidade antes de ampliar o escopo além de personagens.

Espessura, folgas, resistência, retração, autointerseções e precisão dimensional
não foram certificados nesta implementação. Concavidades invisíveis às silhuetas
continuam fora do alcance do método atual.

Referência técnica: a [documentação do Trimesh](https://trimesh.org/trimesh.html)
define `is_watertight` pelo uso duplo das arestas e distingue fechamento,
consistência de orientação e volume. Por isso o relatório identifica o alcance
de cada verificação e mantém as análises ausentes como pendentes.
