# Segmentação Grounding DINO + SAM 2

O adapter `grounded_sam2_v1` mantém as dependências de GPU fora da instalação básica.

## Ambiente de visão

Use um ambiente Python separado com GPU. Instale PyTorch compatível com sua CUDA, depois os projetos oficiais:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e ./GroundingDINO

git clone https://github.com/facebookresearch/sam2.git
pip install -e ./sam2
pip install huggingface_hub
```

Baixe um checkpoint do Grounding DINO e configure:

```bash
export DOLLFORGE_GDINO_CONFIG=/models/GroundingDINO_SwinT_OGC.py
export DOLLFORGE_GDINO_CHECKPOINT=/models/groundingdino_swint_ogc.pth
export DOLLFORGE_SAM2_MODEL=facebook/sam2.1-hiera-base-plus
export DOLLFORGE_VISION_DEVICE=cuda
```

No run, selecione:

```json
{"segmentation_adapter": "grounded_sam2_v1"}
```

Sem essas variáveis, o DollForge continua usando `silhouette_rules_v1`.

## Ontologia inicial

O prompt cobre cabeça, face, cabelo, torso, pelvis, braços superiores, antebraços, mãos,
coxas, canelas, pés, sapatos, camiseta e calça. Todos os resultados permanecem
`needs_review` e entram no mesmo fluxo de correção humana já existente.
