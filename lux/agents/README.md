To add a working agent to this folder, create a file `my_agent` with the following format:

```python
from main import agent_fn, main

class MyAgent:
    def __init__(self, player: str, env_cfg) -> None:
        pass

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        ...
        return actions

if __name__ == "__main__":
    main(agent_fn(MyAgent))
```

This last part for when `__name__ == "__main__"` allows it to work with the `luxai-s3` command line tool like so:
```shell
luxai-s3 agents/my_agent.py agents/my_agent.py --output replay.json
```