import wandb
import sys
from mnist.wandb_project import project_name
from mnist.train_model import train_model

sweep_id = sys.argv[1]

if len(sys.argv) > 2:
    key = sys.argv[2]
    wandb.login(key=key)

print(sweep_id)

wandb.agent(
    sweep_id=sweep_id,
    function=train_model,
    project=project_name,
    count=4,
)

