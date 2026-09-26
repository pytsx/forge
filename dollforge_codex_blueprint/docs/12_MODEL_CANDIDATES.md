# 12 — Candidatos de Modelos e Ferramentas

Este documento é deliberadamente baseado em adapters. Trocar modelos não pode alterar a arquitetura.

## Segmentação

### SAM 2 / SAM 2.1
Uso:
- máscara de objeto;
- refinamento interativo;
- máscara de peças com prompts.

Ponto forte:
feedback humano pode ser convertido em prompts/correções.

Repositório oficial:
https://github.com/facebookresearch/sam2

## Grounding / Detection

### Grounding DINO
Uso:
- localizar semanticamente `head`, `left arm`, `shoe`, etc.;
- gerar proposals que depois são refinadas pelo segmentador.

Repositório oficial:
https://github.com/IDEA-Research/GroundingDINO

## Câmera / Multi-view Geometry

### COLMAP / pycolmap
Uso:
- Structure-from-Motion;
- pose de câmera;
- sparse/dense reconstruction como baseline geométrica.

Repositório oficial:
https://github.com/colmap/colmap

## Image-to-3D

### Hunyuan3D-2 / 2mv
Uso:
- baseline para geração/reconstrução de mesh;
- modo multi-view;
- serviço local;
- integração posterior com Blender.

Repositório oficial:
https://github.com/Tencent-Hunyuan/Hunyuan3D-2

### TRELLIS
Uso:
- baseline image-to-3D;
- experimento multi-image;
- comparação de qualidade por classe de peça.

Repositório oficial:
https://github.com/microsoft/TRELLIS

## Importante

Nenhuma ferramenta acima deve ser considerada “a arquitetura”.

Elas são implementações substituíveis de adapters.

Exemplo:

```python
class SegmentationAdapter(Protocol):
    def predict(self, request: SegmentationRequest) -> SegmentationResult:
        ...
```

## Processo de escolha

Para cada classe de peça:
1. montar benchmark;
2. testar modelos;
3. medir qualidade;
4. medir VRAM/latência;
5. medir taxa de correção humana;
6. escolher adapter padrão;
7. manter fallback.
