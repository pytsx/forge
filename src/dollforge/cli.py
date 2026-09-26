import json
from pathlib import Path
from uuid import UUID

import typer

from dollforge.bootstrap import create_service
from dollforge.domain.models import CreateProject, PipelineConfig, ViewLabel

app = typer.Typer(help="DollForge — engenharia multi-view local")
projects = typer.Typer()
views = typer.Typer()
app.add_typer(projects, name="project")
app.add_typer(views, name="view")


@projects.command("create")
def project_create(name: str, height: float | None = None):
    """Cria um projeto; altura opcional em mm."""
    result = create_service().create_project(CreateProject(name=name, known_height_mm=height))
    typer.echo(result.model_dump_json(indent=2))


@views.command("add")
def view_add(project_id: UUID, label: ViewLabel, image: Path):
    result = create_service().add_view(project_id, label, image.read_bytes())
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def run(project_id: UUID, blender: bool = True):
    service = create_service()
    result = service.new_run(project_id, PipelineConfig(build_blender=blender))
    result = service.execute(result.run_id)
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "failed":
        raise typer.Exit(1)


@app.command()
def replay(run_id: UUID):
    from dollforge.domain.models import RunManifest
    service = create_service()
    previous = service.store.get("run", run_id, RunManifest)
    result = service.new_run(previous.project_id, previous.config, run_id)
    result = service.execute(result.run_id)
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "failed":
        raise typer.Exit(1)


@app.command()
def serve(port: int = 8765):
    """Abre a API e a interface exclusivamente em localhost."""
    import uvicorn
    uvicorn.run("dollforge.api:create_app", factory=True, host="127.0.0.1", port=port)


@app.command()
def schemas(output: Path = Path("schemas/runtime")):
    """Exporta os contratos reais para JSON Schema."""
    from dollforge.domain import models
    output.mkdir(parents=True, exist_ok=True)
    for name in dir(models):
        cls = getattr(models, name)
        if isinstance(cls, type) and issubclass(cls, models.DTO) and cls is not models.DTO:
            (output / f"{name}.schema.json").write_text(
                json.dumps(cls.model_json_schema(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
