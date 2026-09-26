# 15 — Critérios de Aceitação do MVP

O MVP é aprovado quando for possível:

1. criar projeto;
2. enviar 4 vistas;
3. validar qualidade;
4. detectar boneco;
5. segmentar pelo menos:
   - head
   - torso
   - left/right arm
   - pelvis
   - left/right leg
   - footwear
6. corrigir máscara;
7. persistir correção;
8. dar match de peças multi-view;
9. corrigir match;
10. gerar DollGraph;
11. definir escala relativa;
12. gerar meshes baseline;
13. importar meshes no Blender headless;
14. criar `.blend`;
15. armazenar feedback final;
16. reproduzir run usando manifest.

Não é critério do MVP:
- qualidade comercial final;
- single-view;
- treino online;
- geração perfeita de joints ocultos.
