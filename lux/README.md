## Setup Instructions
These instructions are partially copied and modified from the Lux-Design-S3 repository found here: https://github.com/Lux-AI-Challenge/Lux-Design-S3/

It is recommended to use a python package manager like conda/mamba to install the dependencies.
```shell
mamba create -n "lux-s3" "python==3.11"
pip install "git+https://github.com/Lux-AI-Challenge/Lux-Design-S3.git@main#egg=luxai-s3&subdirectory=src"
```
The first command creates a virtual enviornment called "lux-s3" which you should set as your interpreter. 
The second command installs the `src` folder of the [`Lux-Design-S3`](https://github.com/Lux-AI-Challenge/Lux-Design-S3/) GitHub repository as a python package which can be imported in any file with `import luxai_s3`

To verify your installation, you can run a match between two random agents:
```shell
luxai-s3 --help
```
```shell
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```
Then upload the replay.json to the online visualizer here: https://s3vis.lux-ai.org/ (a link on the lux-ai.org website will be up soon)



