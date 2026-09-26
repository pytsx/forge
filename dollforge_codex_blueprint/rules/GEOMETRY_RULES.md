# Geometry Rules

1. Unidade: mm.
2. Z-up.
3. Right-handed.
4. Left/right: personagem.
5. Mesh importada sempre passa por canonical transform.
6. Escala absoluta nunca é inventada silenciosamente.
7. Geometria observada e completada devem ser distinguíveis em metadata.
8. Simetria é constraint, não verdade universal.
9. Connector não deve destruir shape exterior sem autorização.
10. Boolean cutters são objetos separados e nomeados.
11. Não aplicar destructive operation sem guardar source mesh.
12. Topologia para visual e topologia para fabricação podem ser artefatos distintos.
13. Validar normals.
14. Validar manifold.
15. Registrar transform matrix de cada peça.
