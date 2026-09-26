# 20 — API Boundaries

## API pública do sistema

Operações conceituais:

- create project
- add view
- run stage
- get stage result
- submit review
- invalidate descendants
- request regeneration
- build blender
- validate manufacturing
- promote reference
- create dataset snapshot

## Não expor

- paths internos;
- detalhes de GPU;
- objetos Python serializados;
- tabelas específicas de adapters.

## Job semantics

Stages pesados são jobs.

Job:
- queued
- running
- succeeded
- failed
- cancelled
- waiting_for_review

## Idempotency

Uma requisição com:
- mesmos inputs;
- mesma config;
- mesmo model version;

deve reutilizar artefato compatível quando cache estiver habilitado.

## Review API

Feedback nunca edita output antigo em place.

Cria:
- feedback event;
- nova revisão do artifact;
- invalidation record.
