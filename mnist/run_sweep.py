import wandb
from mnist.wandb_project import project_name
from mnist.train_model import train_model

wandb.agent(
    sweep_id='z5t9nxdv',
    function=train_model,
    project=project_name,
)

