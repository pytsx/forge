# 17 — Research Notes

Verificações feitas para orientar o blueprint:

- SAM 2 possui inferência e código de treinamento/fine-tuning, útil para segmentação interativa especializada.
- Grounding DINO fornece detecção open-set guiada por linguagem, útil para propostas de partes.
- COLMAP fornece SfM/MVS e bindings Python, útil para geometria multi-view e câmeras.
- Hunyuan3D-2 possui variantes multi-view, servidor API local e add-on de Blender.
- TRELLIS expõe image-to-3D e uma rota experimental multi-image.

Esses componentes são candidatos, não dependências arquiteturais obrigatórias.

Links oficiais:
- https://github.com/facebookresearch/sam2
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/colmap/colmap
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- https://github.com/microsoft/TRELLIS
