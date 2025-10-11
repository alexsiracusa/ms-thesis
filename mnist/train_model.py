import wandb
import argparse
# from mnist.da

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['adam', 'sgd'], required=True)
args = parser.parse_args()

# Initialize a W&B Run
# with wandb.init('test-project') as run:
#     run.log({'validation_loss':1})