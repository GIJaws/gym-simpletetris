# gym-simpletetris

Note: This project is currently a Work in Progress (WIP). Please be aware that it may contain bugs, incomplete features, or undergo significant changes. Use with caution and feel free to report any issues you encounter.

This is a fork of the original gym-simpletetris project. For basic information about the project, please refer to the [original README](https://github.com/tristanrussell/gym-simpletetris).

## What's New

This fork introduces several updates and improvements to the original gym-simpletetris project:

### 1. Gymnasium Compatibility

The environment has been updated to work with Gymnasium, the latest version of OpenAI Gym. Key changes include:

- Use of `gymnasium` instead of `gym` in imports.
- Updated `step()` method to return 5 values instead of 4, including separate `terminated` and `truncated` flags.
- Updated `reset()` method to return both observation and info dictionary.

### 2. New Usage Example

```python
import gymnasium as gym
import gym_simpletetris
import time

env = gym.make("SimpleTetris-v0", render_mode="human")
obs, info = env.reset()

episode = 0
while episode < 10:
    env.render()  # This line renders the current state
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode {episode + 1} has finished.")
        episode += 1
        obs, info = env.reset()

    time.sleep(0.1)  # Add a small delay to make the game visible

env.close()
```

### 3. Updated Environment Options

The following new options have been added to the environment:

- `render_mode`: Can be set to 'human' or 'rgb_array'. This replaces the old `render()` method options.
- `lock_delay`: Number of steps before a piece locks into place after landing.
- `step_reset`: If True, resets the lock delay when a piece is moved downwards.

### 4. Changes to Info Dictionary

The `info` dictionary returned by `step()` and `reset()` now includes:

- `deaths`: Number of game overs since the environment was created.
- `statistics`: Count of Tetris pieces by type.

### 5. Renderer Updates

The rendering system has been updated to use Pygame, providing better performance and compatibility.

## Installation

For installation instructions, please refer to the original README. The installation process remains the same for this fork.

## Contributing

Contributions to this fork are welcome. Please feel free to submit issues or pull requests.

## License

This project continues to be licensed under the MIT License.

For all other details not mentioned here, please refer to the original gym-simpletetris README.
