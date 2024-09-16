import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from gym_simpletetris.human_play import play_tetris

if __name__ == "__main__":
    play_tetris(render_mode="human", record_actions=False)
