# Security and Privacy Rules

1. Operação local por padrão.
2. Não enviar imagens para serviços externos sem configuração explícita.
3. Registrar licença/origem de todo modelo e dataset.
4. Nunca incluir secrets em logs ou artifacts.
5. Pesos de modelos ficam fora do git.
6. Dados de projetos podem ter ACL futura; não assumir acesso global.
7. Validar arquivos importados.
8. Blender scripts gerados não executam código vindo de metadata não confiável.
9. APIs locais devem suportar bind em localhost.
10. Toda dependência deve ser pinável/reproduzível.
