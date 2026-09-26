# Segmentation Toolkit 2.0

A revisão de máscaras passa a usar a imagem como evidência geométrica, não apenas o gesto do
usuário.

## Edge Engine

O backend possui agora `dollforge.vision.edges` com:
- mapa Sobel normalizado;
- region growing limitado por bordas;
- continuidade de cor;
- score de aderência da máscara às arestas.

`contour_rules_v2` usa esse motor para tentar separar regiões que estão conectadas na
silhueta mas possuem uma borda interna visível, como roupa/corpo e perna/calçado.

## Ferramentas do editor

### Região inteligente
Clique dentro da peça. A seleção cresce por pixels semelhantes e para em bordas fortes.

### Varinha
Versão mais conservadora da Região inteligente: menor tolerância cromática e maior
sensibilidade a bordas.

### Laço magnético
Marque poucos pontos próximos ao contorno. Cada trecho procura a aresta mais forte próxima
da linha indicada. Use **Fechar laço** ou duplo clique para aplicar.

### Refinar borda
Reposiciona os limites atuais da máscara para gradientes próximos, preservando a forma
geral da seleção.

### Expandir / Contrair
Operações morfológicas pequenas para corrigir folgas sistemáticas.

### Preencher
Preenche buracos fechados dentro da máscara.

### Visualizar bordas
Sobrepõe o edge map sobre a imagem para tornar explícito o que o algoritmo está usando.

## Controles

- **Similaridade** controla o quanto a seleção aceita mudança de cor.
- **Bordas** controla o quanto gradientes da imagem funcionam como barreira.
- `Alt + clique` com Região/Varinha remove a região encontrada em vez de adicionar.

## Princípio de produto

Essas ferramentas não substituem a máscara humana. Elas aceleram a correção e o resultado
continua entrando no mesmo fluxo append-only de revisão. Ao salvar, a propagação
multi-view introduzida anteriormente continua recalculando apenas vistas não revisadas.
