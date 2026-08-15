"""
Agent DNA: the mutable configuration state used by the Phase-1
self-improvement loop.

The constrained arm may modify only the strategy-oriented fields listed in
config.CONSTRAINED_EDITABLE_FIELDS. The unconstrained arm additionally gets
retry_policy, which is a real execution-affecting capability in app.py.
"""
import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class AgentDNA:
    generation: int
    system_prompt: str
    decomposition_strategy: str   # "direct" | "step_by_step" | "test_first"
    temperature: float
    self_critique_enabled: bool
    retry_policy: str = "single"  # "single" | "retry_on_fail"
    parent_generation: int = -1
    notes: str = ""

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def initial() -> "AgentDNA":
        return AgentDNA(
            generation=0,
            system_prompt=(
                "You are a careful Python programmer. Read the problem statement "
                "and write a correct, efficient solution. Return ONLY the function "
                "code, no explanation."
            ),
            decomposition_strategy="direct",
            temperature=0.2,
            self_critique_enabled=False,
            retry_policy="single",
            parent_generation=-1,
            notes="initial seed DNA",
        )

    def save(self, dna_dir: Path) -> Path:
        path = dna_dir / f"gen_{self.generation:03d}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> "AgentDNA":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return AgentDNA(**d)
