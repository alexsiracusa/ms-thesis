import wandb
import sys
from mnist.dataset_features.wandb_project import project_name
from mnist.dataset_features.wandb_run import wandb_run

sweep_id = sys.argv[1]

if len(sys.argv) > 2:
    key = sys.argv[2]
    wandb.login(key=key)

print(sweep_id)

wandb.agent(
    sweep_id=sweep_id,
    function=wandb_run,
    project=project_name,
)

