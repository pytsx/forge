import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dollforge.contracts import MeshCandidate
from dollforge.errors import DependencyUnavailable, DomainError
from dollforge.storage import canonical


def find_blender() -> str | None:
    configured = os.environ.get("DOLLFORGE_BLENDER")
    if configured:
        return configured if Path(configured).is_file() else None
    found = shutil.which("blender")
    if found:
        return found
    base = Path(os.environ.get("PROGRAMFILES", "/nonexistent")) / "Blender Foundation"
    candidates = sorted(base.glob("Blender */blender.exe"), reverse=True)
    return str(candidates[0]) if candidates else None


class HeadlessBlender:
    def __init__(self):
        self.executable = find_blender()
        self.model_version = "unavailable"
        if self.executable:
            result = subprocess.run([self.executable, "--version"], capture_output=True,
                                    text=True, timeout=30)
            self.model_version = result.stdout.splitlines()[0]

    def build(self, meshes: list[MeshCandidate], unit: str) -> tuple[bytes, str]:
        if not self.executable:
            raise DependencyUnavailable("Blender não encontrado. Configure DOLLFORGE_BLENDER.")
        worker = Path(__file__).resolve().parents[1] / "workers" / "blender_build.py"
        with tempfile.TemporaryDirectory(prefix="dollforge-") as directory:
            root = Path(directory)
            request = root / "request.json"
            destination = root / "assembly.blend"
            request.write_bytes(canonical({"meshes": [m.model_dump(mode="json") for m in meshes],
                                           "unit": unit}))
            result = subprocess.run(
                [self.executable, "--background", "--factory-startup", "--python-exit-code", "1",
                 "--python", str(worker), "--", str(request), str(destination)],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode or not destination.exists():
                raise DomainError("Blender falhou ao gerar o arquivo. Verifique a instalação e os meshes.")
            return destination.read_bytes(), self.model_version
