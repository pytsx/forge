# Imperative Quality Limit Loop

O DollForge não pode mais avançar apenas porque um especialista produziu uma saída.

Cada etapa que possui geração adaptativa precisa:

```text
GERAR
  ↓
MEDIR
  ↓
PROVAR O LIMITE
  ├─ PASS → avançar
  └─ FAIL
       ↓
    especialista lê o erro
       ↓
    ajusta parâmetros de geração
       ↓
    gera novamente
```

O limite é imutável durante as tentativas. O especialista pode alterar parâmetros da
geração, mas nunca reduzir a meta para transformar uma falha em aprovação.

## Estados

Uma etapa quality-gated termina somente em:

- `passed`: uma tentativa atingiu todos os limites;
- `retrain_candidate`: o orçamento de tentativas terminou sem atingir a prova.

Não existe "aceitar o melhor mesmo reprovado" no fluxo normal.

## Tentativas

Default:

```text
quality_max_attempts = 3
```

Cada tentativa registra:

- parâmetros usados;
- métricas observadas;
- limites exigidos;
- hard failures;
- score agregado.

O histórico completo é persistido como `LimitTrace`.

## Autoajuste

### Segmentação

O especialista pode ajustar:

- foreground threshold;
- edge threshold;
- color tolerance;
- Grounding DINO box threshold;
- Grounding DINO text threshold.

A regra observa:

- aderência da máscara às bordas;
- confiança;
- cobertura semântica.

Aderência à borda é hard metric.

### Matching

O especialista `multisignal_v1` ajusta:

- distância máxima de associação.

A prova observa:

- confiança média;
- cobertura entre vistas.

### Volumetria

O especialista ajusta:

- resolução do grid;
- soft support threshold.

A prova observa:

- mean reprojection IoU;
- outside silhouette error;
- maximum boundary overshoot.

Máscaras confirmadas pelo humano são hard constraints e têm prioridade nas métricas de
não extrapolação.

## Regra de borda

Para uma vista confirmada:

```text
project(volume) ⊆ silhouette + raster_tolerance
```

O especialista pode aumentar resolução ou remover geometria. Ele não pode aumentar a
tolerância para "passar na prova".

## Ponto de retreino

Quando todas as tentativas falham, é criado um `TrainingSignal` com:

- estágio;
- especialista;
- inputs;
- trace completo;
- melhor tentativa;
- parâmetros da melhor tentativa;
- métricas ainda reprovadas.

`training_authorized = false`.

Isso é proposital. O run pode aprender por ajuste de parâmetros imediatamente, mas
alteração de pesos continua sendo um processo offline, curado e versionado.

## Fail closed

As propriedades abaixo são invariantes da configuração normal:

```text
quality_loop_enabled = true
quality_fail_closed = true
retrain_on_limit_exhaustion = true
```

Se o limite não é atingido, o run entra em `waiting_for_review` no estágio que falhou e
as etapas dependentes não são executadas.

## Cache

Etapas quality-gated não reutilizam silenciosamente o resultado do cache. A prova é
executada novamente para que cada run possua seu próprio `LimitTrace`.

## Próximos especialistas

O mesmo contrato será aplicado aos próximos adapters:

- depth;
- surface normals;
- SDF refinement;
- connector inference;
- mesh repair;
- manufacturing validation.

Cada especialista deverá declarar:

1. parâmetros que pode modificar;
2. métricas que precisa atingir;
3. estratégia de ajuste baseada no tipo de erro;
4. orçamento de tentativas;
5. assinatura de falha que gera sinal de retreino.
