# Estrutura de código sugerida

```text
src/dollforge/
├── api/
├── domain/
│   ├── ontology.py
│   ├── models.py
│   └── enums.py
├── contracts/
├── orchestration/
│   ├── dag.py
│   └── stages/
├── adapters/
│   ├── segmentation/
│   ├── detection/
│   ├── camera/
│   ├── reconstruction/
│   ├── retrieval/
│   └── blender/
├── geometry/
├── joints/
├── manufacturing/
├── knowledge/
├── feedback/
├── training/
├── evaluation/
└── storage/
```

## Interface conceitual de stage

```python
from typing import Protocol, Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")

class Stage(Protocol, Generic[I, O]):
    name: str
    version: str

    def run(self, input: I, context: "RunContext") -> O:
        ...
```

## Interface conceitual de adapter

```python
class SegmentationAdapter(Protocol):
    model_id: str
    model_version: str

    def predict(self, request: SegmentationRequest) -> SegmentationResult:
        ...
```

## Requisito

Stages não importam implementações concretas diretamente.

Use dependency injection/config registry.
