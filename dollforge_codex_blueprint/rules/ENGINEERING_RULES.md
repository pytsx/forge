# Engineering Rules

1. Python tipado.
2. `pydantic` para DTOs/schemas runtime.
3. `Protocol`/ABC para adapters.
4. Sem strings mágicas para labels.
5. Sem paths absolutos dentro de módulos.
6. Configuração por arquivo/env.
7. Funções puras para transforms quando possível.
8. Logs estruturados com IDs.
9. Erros de domínio próprios.
10. Retries só em falhas transitórias.
11. Idempotência em stages.
12. Cache baseado em hash de input+config+model.
13. Não esconder exceptions de geometry.
14. Feature flags para modelos experimentais.
15. Toda integração externa fica atrás de adapter.
