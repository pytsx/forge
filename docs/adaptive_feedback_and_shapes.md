# Adaptação supervisionada em tempo real

Esta versão melhora dois pontos observados no primeiro protótipo: máscaras retangulares e
geometria 3D excessivamente genérica.

## Segmentação por contorno

O baseline local padrão passa de `silhouette_rules_v1` para `contour_rules_v2`.

As regiões semânticas deixam de ser retângulos fixos. O sistema usa priors curvos de
bonecos e os intersecta com a silhueta real da imagem. Isso produz máscaras que acompanham
melhor cabeça, torso, braços, pernas e calçados mesmo sem GPU.

Grounding DINO + SAM 2 continua sendo a rota de maior capacidade quando configurada.

## Correção humana como prior imediato

Ao corrigir uma máscara:

1. a versão humana é preservada como `human_edited`;
2. o sistema extrai posição vertical e perfil de largura da forma corrigida;
3. procura a mesma classe/lado nas outras vistas;
4. recalcula somente observações ainda automáticas;
5. intersecta a proposta com o foreground específico de cada vista;
6. grava novas máscaras como `model_inferred`;
7. mantém essas máscaras como `needs_review`;
8. nunca sobrescreve uma vista já corrigida ou aprovada por humano;
9. invalida matching, DollGraph e reconstrução dependentes.

Isso é **adaptação online do projeto**, não fine-tuning dos pesos. A resposta muda em tempo
real, permanece explicável e reversível e gera exemplos úteis para treino offline futuro.

## Reconstrução 3D por templates semânticos

O baseline padrão passa para `doll_templates_multiview_v2`.

Cada classe recebe um prior geométrico distinto:
- cabeça: superelipsoide arredondado com profundidade mínima coerente;
- torso: volume cônico/arredondado, mais próximo de um corpo/camiseta;
- pelvis: volume achatado e arredondado;
- braços/pernas: membros robustos;
- mãos: volumes compactos;
- pés/calçados: base mais larga e achatada.

As dimensões continuam vindo das observações multi-view. A geometria ainda não substitui
um reconstrutor neural de alta fidelidade, mas deixa de representar todas as peças pela
mesma primitiva.
