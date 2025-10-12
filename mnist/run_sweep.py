import wandb
from mnist.wandb_project import project_name
from mnist.train_model import train_model

wandb.agent(
    sweep_id='l1r0aps1',
    function=train_model,
    project=project_name,
)

