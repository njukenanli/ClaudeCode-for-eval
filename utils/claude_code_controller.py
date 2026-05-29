import json, os, datetime
from functools import partial
import time
from typing import Any, Optional, TypedDict
from utils.docker_runtime import Runtime
from utils.type import SWEBenchInstance

def logger(base_dir: str, text: str) -> None:
    os.makedirs(f"{base_dir}", exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{base_dir}/debug.log", mode="a") as f:
        print(f"\n\n{current_time}\n{text}\n", file=f)

class SWEbenchInput(TypedDict):
    instance_id: str
    model_patch: str
    model_name_or_path: str

class ClaudeCodeMessage(TypedDict):
    type: str
    content: list[Any]

class ClaudeCodeStep(TypedDict):
    type: str
    message: ClaudeCodeMessage
    parent_tool_use_id: Optional[str | None]

ClaudeCodeTraj = list[ClaudeCodeStep]

class AgentResult(SWEbenchInput):
    trajectory: ClaudeCodeTraj


class ClaudeController:
    system_prompt = """You are an expert software engineer solving swebench bug fixing tasks."""

    def __init__(
        self, 
        image: str, 
        instance_id: str, 
        base_dir: str, 
        tools: set[str], 
        user_prompt: str, 
        endpoint: str, 
        api_key: str, 
        working_dir: str
    ) -> None:
        self.image = image
        self.base_dir = base_dir
        self.endpoint = endpoint
        self.api_key = api_key
        self.working_dir = working_dir
        self.container: Runtime = self.init_container(self.image, instance_id, working_dir)
        self.tools: str = ",".join([i.strip() for i in tools if i.strip()])
        assert "{description}" in user_prompt
        self.user_prompt: str = user_prompt
        return

    def init_container(self, image: str, instance_id: str, working_dir: str) -> Runtime:
        container = Runtime.start_session(
            image,
            instance_id,
            log_function=partial(logger, base_dir=self.base_dir),
            working_dir=working_dir,
        )
        container.send_command("git config --global --add safe.directory /testbed")
        container.send_command("apt-get install curl -y")
        container.send_command("curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.89")
        time.sleep(10)
        container.send_command('alias claude="$HOME/.local/bin/claude"')

        container.send_command(f"export ANTHROPIC_BASE_URL={self.endpoint}")
        container.send_command(f"export ANTHROPIC_AUTH_TOKEN={self.api_key}")
        container.send_command("export IS_SANDBOX=1")
        return container

    def _run_cli(self, instance: SWEBenchInstance, max_step: int, timelimit: int) -> ClaudeCodeTraj:
        # prepare prompt safely: write it to a file inside the container using a single-quoted heredoc
        # directly applying prompt for heredoc may raise error for windows line ending \r\n
        prompt_text = self.user_prompt.format(description=instance["problem_statement"].replace('"""', "'''"))
        # choose a simple filename and a heredoc delimiter unlikely to collide
        heredoc_cmd = "cat > /tmp/cc_prompt.txt <<'CC_PROMPT'\n" + prompt_text + "\nCC_PROMPT\n"
        self.container.send_command(heredoc_cmd)

        # run claude reading the prompt from the file to avoid shell interpolation issues
        claude_cmd = f'claude -p "$(cat /tmp/cc_prompt.txt)" --system-prompt "{self.system_prompt}" --tools {self.tools} --max-turns {max_step}  --dangerously-skip-permissions  --output-format json  --verbose'
        res = self.container.send_command(claude_cmd, timelimit * 60)
        traj = [i for i in res.output.splitlines() if "session_id" in i]
        if len(traj) == 0:
            raise TimeoutError(f"Claude Code fails to complete the task in {timelimit} min...")
        traj: ClaudeCodeTraj = json.loads(traj[0])
        # self.container.send_command("cat /tmp/hook.out")
        return traj

    def run_instance(
        self, instance: SWEBenchInstance, max_step: int = 40, timelimit: int = 120
    ) -> AgentResult:
        """
        timelimit: in minute
        """
        traj = self._run_cli(instance, max_step, timelimit)
        self.container.send_command(f"cd {self.working_dir}")
        solution_patch = self.container.send_command("git --no-pager diff HEAD  --text").output
        solution_patch = solution_patch.replace("git --no-pager diff HEAD  --text\n", "")
        return_value: AgentResult = {
            "instance_id": instance["instance_id"],
            "model_patch": solution_patch,
            "model_name_or_path": "cc",
            "trajectory": traj,
        }
        return return_value

    def __del__(self):
        if hasattr(self, "container"):
            self.container.cleanup()
