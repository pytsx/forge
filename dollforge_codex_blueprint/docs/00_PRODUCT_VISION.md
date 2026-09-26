# 00 — Visão do Produto

## Problema

Reconstrução 3D genérica tende a produzir uma malha visual, não um produto físico modular.

Bonecos possuem semântica própria:
- cabeça, torso, pelvis, membros;
- simetria;
- articulações;
- encaixes;
- roupas;
- peças removíveis;
- limites mecânicos;
- tolerâncias de fabricação;
- identidade estética.

O sistema deve aprender esse domínio.

## Resultado alvo

Dadas imagens multi-view de um boneco, gerar:

1. projeto semântico;
2. peças individualizadas;
3. geometria alinhada e coerente;
4. articulações e encaixes plausíveis;
5. assembly 3D;
6. arquivos Blender editáveis;
7. exports para fabricação;
8. relatório de validação;
9. pacote de feedback para aprendizado.

## Evolução

### Fase A — Multi-view supervisionado
Usar várias vistas e intervenção humana frequente.

### Fase B — Multi-view altamente automático
Intervenção apenas em baixa confiança.

### Fase C — Few-view
2–3 imagens + knowledge priors.

### Fase D — Single-view
Uma imagem + conhecimento aprendido do universo de bonecos.

A transição para single-view só deve ocorrer quando o sistema tiver priors estruturais e biblioteca suficientemente maduros.
