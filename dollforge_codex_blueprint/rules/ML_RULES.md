# ML Rules

1. Não treinar em outputs automáticos não aprovados.
2. Registrar model name, checksum e config.
3. Não promover modelo só por loss menor.
4. Benchmark fixo obrigatório.
5. Avaliar por classe de peça.
6. Separar train/validation/test por personagem/projeto, evitando leakage entre vistas do mesmo objeto.
7. Deduplicar referências.
8. Registrar distribuição de classes.
9. Calibrar confidence quando usada para autoaprovação.
10. Tratar OOD explicitamente.
11. Manter baseline simples.
12. Fine-tuning deve ser reproduzível.
13. Adapters precisam aceitar versão de modelo.
14. Dataset manifests são imutáveis.
15. Human feedback precisa de provenance.
