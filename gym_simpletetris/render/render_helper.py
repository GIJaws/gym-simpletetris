from gym_simpletetris.render.human_renderer import HumanRenderer
from gym_simpletetris.render.array_renderer import ArrayRenderer


def create_renderer(width, height, buffer_height, visible_height, obs_type, render_mode, window_size, render_fps):
    if render_mode == "human":
        return HumanRenderer(
            width=width,
            height=height + buffer_height,
            obs_type=obs_type,
            block_size=window_size // visible_height,
            fps=render_fps,
            visible_height=visible_height,
        )
    elif render_mode in ["rgb_array"]:
        return ArrayRenderer(
            width=width,
            height=height + buffer_height,
            visible_height=visible_height,
            obs_type=obs_type,
        )
    else:
        raise ValueError(f"Unsupported render mode: {render_mode}")
