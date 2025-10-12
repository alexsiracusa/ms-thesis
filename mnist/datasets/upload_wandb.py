import wandb
from mnist.datasets import datasets
from mnist.wandb_project import project_name


run = wandb.init(project=project_name, name="dataset-upload")
api = wandb.Api()
existing = [coll.name for coll in api.artifact_type(type_name='dataset', project=project_name).collections()]

for dataset in datasets:
    if dataset in existing:
        continue

    artifact = wandb.Artifact(
        name=dataset,
        type="dataset",
    )

    artifact.add_file(f"./parquets/{dataset}/train.parquet")
    artifact.add_file(f"./parquets/{dataset}/test.parquet")
    wandb.log_artifact(artifact)
    artifact.wait()
