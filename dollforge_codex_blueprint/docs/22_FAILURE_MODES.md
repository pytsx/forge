# 22 — Catálogo de Failure Modes

## Percepção
- left/right invertido;
- roupa confundida com corpo;
- mão fundida ao braço;
- cabelo fundido ao torso;
- oclusão;
- sombra interpretada como geometria.

## Multi-view
- duas peças diferentes agrupadas;
- uma peça dividida em duas identities;
- vista frontal espelhada;
- pose diferente entre fotos;
- escala inconsistente;
- câmera estimada incorretamente.

## Reconstrução
- peça achatada;
- volumes inventados;
- detalhe excessivo;
- ruído de superfície;
- assimetria indevida;
- mesh non-manifold.

## Joints
- joint semanticamente errado;
- eixo deslocado;
- range impossível;
- connector frágil;
- folga inadequada;
- socket perfura parede.

## Assembly
- gaps;
- interpenetração;
- mãos/feet desalinhados;
- centro de articulação incorreto;
- pose neutra inconsistente.

## Learning
- leakage;
- feedback loop;
- gold dataset contaminado;
- duplicatas;
- drift de style family;
- overfitting a um personagem.

## Resposta do sistema

Cada failure mode deve ter:
- code;
- severity;
- detector quando possível;
- suggested remediation;
- stage owner.
