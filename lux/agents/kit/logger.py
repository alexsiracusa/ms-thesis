import json
import numpy as np
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def log_actions(file_name, agent, obs, actions, remainingOverageTime):
    player_info = dict()
    player_info['obs'] = obs
    player_info['remainingOverageTime'] = remainingOverageTime
    player_info['player'] = agent.player
    player_info['info'] = {'env_cfg': agent.env_cfg}
    player_info['actions'] = actions

    _append_to_json_array(file_name, player_info)

def _append_to_json_array(filename, new_element):
    # Check if file exists and is non-empty
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, 'w') as f:
            json.dump([new_element], f, cls=NumpyEncoder, indent=2)
        return

    # Open file in r+ mode (read+write)
    with open(filename, 'r+') as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1

        # Move to just before the closing bracket
        while pos > 0:
            f.seek(pos)
            char = f.read(1)
            if char == ']':
                pos -= 1
                f.seek(pos)
                char = f.read(1)
                if char == '[':
                    # Empty array case: insert directly
                    f.seek(pos + 1)
                    f.write(json.dumps(new_element, cls=NumpyEncoder, indent=2) + ']')
                    return
                else:
                    # Backtrack until we find the last non-whitespace character before ']'
                    while char.isspace() or char == '\n':
                        pos -= 1
                        f.seek(pos)
                        char = f.read(1)
                    f.seek(pos + 1)
                    f.write(',' + json.dumps(new_element, cls=NumpyEncoder, indent=2) + ']')
                    return
