from typing import Optional, TypedDict

class SWEBenchInstance(TypedDict):
    instance_id: str
    source: str
    repo: str
    base_commit: str
    problem_statement: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    test_cmds: str
    patch: str
    test_patch: str
    docker_image: str
    working_dir: Optional[str]