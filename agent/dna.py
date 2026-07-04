"""
Agent 'DNA': the mutable state that represents what the agent currently is.
This is the ONLY thing the self-improvement loop is allowed to modify in
Phase 1. No raw source code is executed as a patch target here — this keeps
the whole loop safe to run on a personal laptop with no sandboxing beyond
subprocess timeouts for the *generated solutions*, not for the agent itself.

Phase 2 (see README "Roadmap") extends this to real module-level code
patching once the measurement harness below is validated.
"""
import dataclasses
import json
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class AgentDNA:
    generation: int
    system_prompt: str
    decomposition_strategy: str   # "direct" | "step_by_step" | "test_first"
    temperature: float
    self_critique_enabled: bool
    confidence_threshold: float = 0.5   # unconstrained-arm-only field
    retry_policy: str = "single"        # unconstrained-arm-only field
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
            parent_generation=-1,
            notes="initial seed DNA",
        )

    def save(self, dna_dir: Path) -> Path:
        path = dna_dir / f"gen_{self.generation:03d}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @staticmethod
    def load(path: Path) -> "AgentDNA":
        d = json.loads(Path(path).read_text())
        return AgentDNA(**d)
