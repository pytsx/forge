# Dependency Rules

## Domain layer
Não pode depender de:
- Blender;
- PyTorch;
- vendor models;
- FastAPI.

## Orchestration
Pode depender dos contracts/interfaces, não de implementações concretas.

## Adapters
São o único lugar autorizado para dependências específicas de vendors/models.

## Blender
Código Blender fica isolado para evitar conflito de Python environment.

## Training
Não pode ser importado pelo runtime de inference.

## Review UI
Nunca acessa banco diretamente; usa API/service layer.
