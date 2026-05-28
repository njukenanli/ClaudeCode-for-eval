from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import yaml
from server.server import Server
from utils.claude_code_controller import ClaudeController, AgentResult
from utils.type import SWEBenchInstance

class Agent:
    def __init__(self,
                config_path: str,
                run_id: str = "debug"):
        self.run_id = run_id
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.model_args, self.agent_args = config["model"], config["agent"]
        self.model_name = self.model_args["model"].replace("/", "__")
        self.log_dir = f"logs/{self.model_name}/{self.run_id}"
    
    def run_instance(self, 
                     instance: SWEBenchInstance, 
                     working_dir: str = "/testbed"
                    ) -> str:
        '''
        input: swe bench instance dict, docker image working_dir default to /testbed
        output: save prediction patch, agent trajectory to logs dir and returns patch
        '''
        patch: str = ""
        instance_id = instance["instance_id"]
        out_dir = f"{self.log_dir}/{instance_id}"
        if os.path.exists(f"{out_dir}/traj.json") and os.path.exists(f"{out_dir}/patch.diff"):
            with open(f"{out_dir}/patch.diff") as f:
                patch=f.read()
            return patch

        server = Server()
        if "azure" in self.model_args["model"]:
            base_url, token = server.start_from_azure_openai(**self.model_args)
        else:
            base_url, token = server.start_from_api_key(**self.model_args)
        try:
            self.cc: ClaudeController = ClaudeController(
                instance["docker_image"],
                instance_id,
                out_dir,
                self.agent_args["tools"],
                self.agent_args["user_prompt"],
                base_url,
                token,
                working_dir
            )

            res: AgentResult = self.cc.run_instance(
                instance, self.agent_args["max_step"], self.agent_args.get("timeout", 120)
            )

            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/traj.json", "w") as f:
                json.dump(res["trajectory"], f, indent=2)
            with open(f"{out_dir}/patch.diff", "w") as f:
                f.write(res["model_patch"])
            patch = res["model_patch"]
        except TimeoutError as e:
            print(instance["instance_id"], e, flush=True)
        
        server.stop()
        return patch

    def run_instances(self, 
                      instances: list[SWEBenchInstance], 
                      ):
        workers = self.agent_args.get("workers", 1)
        predictions: dict[str, dict[str, str]] = {}

        def run(instance: SWEBenchInstance) -> tuple[str, str]:
            patch = self.run_instance(instance, instance.get("working_dir", "/testbed"))
            return instance["instance_id"], patch

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run, instance) for instance in instances]
            for future in as_completed(futures):
                instance_id, patch = future.result()
                predictions[instance_id] = {
                    "instance_id": instance_id,
                    "model_patch": patch,
                }
                with open(f"{self.log_dir}/preds.json", "w") as f:
                    json.dump(predictions, f, indent=True)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    dataset: list[SWEBenchInstance]
    if args.dataset.endswith("jsonl"):
        with open(args.dataset) as f:
            dataset = [json.loads(i) for i in f]
    else:
        from datasets import load_dataset
        dataset = load_dataset(args.dataset, split=args.split)
    agent = Agent(args.config, args.run_id)
    agent.run_instances(dataset)