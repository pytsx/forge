# Matching multi-view

O adapter `multisignal_v1` substitui a regra simples `classe + lado` como padrão.

Ele usa restrições fortes:
- mesma classe semântica;
- lado compatível;
- nunca associa duas observações da mesma vista.

E combina sinais contínuos:
- posição vertical normalizada;
- escala relativa do bbox;
- aspect ratio;
- preenchimento da máscara;
- média e desvio de cor da região mascarada.

O algoritmo é determinístico e mantém provenance.

## Por que não embedding neural ainda?

Esta etapa cria primeiro o contrato estável e um baseline mensurável. O descritor pode ser
substituído futuramente por DINOv2/CLIP ou um embedding treinado especificamente em peças
de bonecos sem alterar o `MatchingRequest`, o `DollGraph` ou a interface de revisão.

## Limitação

Sem câmera calibrada, sinais de posição lateral e perspectiva ainda são fracos. O próximo
salto é combinar este matcher com pose de câmera/epipolaridade.
