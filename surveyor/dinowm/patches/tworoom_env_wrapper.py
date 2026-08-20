"""Deployed as dino_wm/env/tworoom/tworoom_env_wrapper.py.

TwoRoom eval env for the DINO-WM closed-loop battery: a dependency-free
re-implementation of stable_worldmodel.envs.two_room.env.TwoRoomEnv's geometry,
rendering and collision, plus the vector-env contract dino_wm's PlanWorkspace /
PlanEvaluator require (prepare / rollout / step_multiple / eval_state /
sample_random_init_goal_states / update_env).

Why vendored rather than imported: stable_worldmodel lives in the le-wm venv
(py3.11) and pulls gymnasium + swm_spaces; dino_wm runs py3.9. Every constant
below is copied verbatim from that file, and the layout is pinned to the
variation space's init_value because the dataset uses exactly one layout
(measured over 400 episodes of tworoom.h5: door centre (112, 49) in every
episode, one door, vertical wall). validate_tworoom_env.py checks the vendored
renderer and dynamics against the recorded pixels before any battery runs.

GOAL SEMANTICS. The env never renders the target (measured: zero green pixels
in the data), which is what made TwoRoom ill-posed for goal-free drafting on
LeWM. DINO-WM plans toward a goal IMAGE, and under goal_source='random_state'
that image is this env rendered with the AGENT AT the goal position -- so the
destination is fully observable in obs_g and the task is well posed here. The
init/goal pair is drawn in OPPOSITE rooms, matching the dataset (the env's own
_constrain_target_by_min_steps encodes the same intent).
"""
import gym
import numpy as np
import torch

from utils import aggregate_dct

# ---- geometry, copied verbatim from TwoRoomEnv -----------------------------
IMG_SIZE = 224
BORDER_SIZE = 14
PADDING = 14
WALL_CENTER = 112
MAX_DOOR = 3

# ---- layout: the variation-space init_value, which is what the dataset used
WALL_AXIS = 1            # vertical wall at x = WALL_CENTER
WALL_THICKNESS = 10
NUM_DOORS = 1
DOOR_POSITION = 49.0     # centre coord along the wall
DOOR_SIZE = 14.0         # half-extent, in pixels
AGENT_RADIUS = 7.0
AGENT_SPEED = 5.0

COLOR_BG = (255, 255, 255)
COLOR_WALL = (0, 0, 0)
COLOR_DOOR = (255, 255, 255)
COLOR_AGENT = (255, 0, 0)

SUCCESS_RADIUS = 16.0    # TwoRoomEnv.step's own `terminated` criterion
ENV_ACTION_DIM = 2


class TwoRoomEnvWrapper(gym.Env):
    """TwoRoom under the vector-env contract dino_wm's planner expects.

    A dependency-free re-implementation, so the closed-loop battery runs
    without stable_worldmodel installed alongside dino_wm.
    """
    metadata = {"render_modes": ["rgb_array"], "render.modes": ["rgb_array"]}
    reward_range = (0.0, 0.0)

    def __init__(self, **kwargs):
        self.action_dim = ENV_ACTION_DIM
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        y = torch.arange(IMG_SIZE, dtype=torch.float32)
        x = torch.arange(IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing="ij")

        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.goal_position = torch.zeros(2, dtype=torch.float32)
        self._rng = np.random.RandomState(0)
        self._wall_mask, self._door_mask = self._wall_and_door_masks()
        self._base_img = self._render_base()

    # ---------------- dino_wm vector-env contract ----------------

    def seed(self, seed=None):
        """Store a seeded RNG and return `[seed]`, as gym's contract expects.

        Nothing reads the stored RNG: sample_random_init_goal_states builds
        its own from the seed it is handed, so seeding is per call rather
        than stateful.
        """
        self._rng = np.random.RandomState(None if seed is None else int(seed))
        return [seed]

    def update_env(self, env_info):
        """No-op: the dataset uses a single fixed layout (verified over 400
        episodes), so there is nothing per-episode to match."""
        return

    def eval_state(self, goal_state, cur_state):
        """Success and agent-to-goal distance, on the env's own 16px radius."""
        goal_state = np.asarray(goal_state, dtype=np.float32).reshape(-1)
        cur_state = np.asarray(cur_state, dtype=np.float32).reshape(-1)
        state_dist = float(np.linalg.norm(goal_state[:2] - cur_state[:2]))
        return {
            "success": bool(state_dist < SUCCESS_RADIUS),
            "state_dist": state_dist,
        }

    def sample_random_init_goal_states(self, seed):
        """Draw (init, goal) agent positions in OPPOSITE rooms.

        Matches the dataset: every recorded episode has agent and target on
        opposite sides of the wall, which is the task TwoRoom exists to pose
        (the route must pass through the door)."""
        rng = np.random.RandomState(int(seed))
        left_first = rng.rand() < 0.5
        init = self._sample_in_room(rng, left=left_first)
        goal = self._sample_in_room(rng, left=not left_first)
        return init.astype(np.float32), goal.astype(np.float32)

    def prepare(self, seed, init_state):
        """Reset with a controlled agent position. Returns (obs, state)."""
        self.seed(seed)
        self.agent_position = torch.as_tensor(
            np.asarray(init_state, dtype=np.float32)[:2], dtype=torch.float32)
        obs = self._get_obs()
        return obs, self._get_state()

    def step(self, action):
        """Apply one clipped action and return (obs, reward, done, info).

        Reward is always 0 and done always False: the planner scores episodes
        through eval_state, not the env's own signal.
        """
        action_t = torch.as_tensor(np.asarray(action, dtype=np.float32),
                                   dtype=torch.float32)
        action_t = torch.clamp(action_t, -1.0, 1.0)
        pos_next = self.agent_position + action_t * AGENT_SPEED
        self.agent_position = self._apply_collisions(self.agent_position, pos_next)

        dist = float(torch.norm(self.agent_position - self.goal_position))
        obs = self._get_obs()
        info = {
            "state": self._get_state(),
            "proprio": self.agent_position.numpy().copy(),
            "distance_to_target": dist,
        }
        return obs, 0.0, False, info

    def step_multiple(self, actions):
        """Apply a sequence of actions, returning each step's output stacked."""
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """seed: int; init_state: (2,); actions: (T, 2).
        obses: dict of (T+1, ...); states: (T+1, 2)."""
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, np.stack(states)

    def render(self):
        """Render the current agent position to an rgb_array frame."""
        return self._render_agent(self.agent_position)

    # ---------------- internals ----------------

    def _get_state(self):
        return self.agent_position.numpy().copy()

    def _get_obs(self):
        return {
            "visual": self._render_agent(self.agent_position),
            "proprio": self.agent_position.numpy().copy(),
        }

    def _sample_in_room(self, rng, left):
        """Uniform over the valid interior of one room (border- and
        wall-clearance respected, exactly as _apply_collisions clamps)."""
        lo = BORDER_SIZE + AGENT_RADIUS
        hi = IMG_SIZE - BORDER_SIZE - AGENT_RADIUS
        half = WALL_THICKNESS // 2
        if left:
            x_lo, x_hi = lo, WALL_CENTER - half - AGENT_RADIUS - 0.5
        else:
            x_lo, x_hi = WALL_CENTER + half + AGENT_RADIUS + 0.5, hi
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(lo, hi)
        if WALL_AXIS == 1:
            return np.array([x, y], dtype=np.float32)
        return np.array([y, x], dtype=np.float32)

    def _gaussian_dot(self, pos_xy, radius):
        dx = self.grid_x - float(pos_xy[0])
        dy = self.grid_y - float(pos_xy[1])
        dist2 = dx * dx + dy * dy
        std = max(1e-6, float(radius))
        dot = torch.exp(-dist2 / (2.0 * std * std))
        m = dot.max()
        if m > 0:
            dot = dot / m
        return dot

    def _wall_and_door_masks(self):
        H = W = IMG_SIZE
        half = WALL_THICKNESS // 2
        if WALL_AXIS == 1:
            wall_stripe = (self.grid_x >= (WALL_CENTER - half)) & (
                self.grid_x <= (WALL_CENTER + half))
            door_span = torch.zeros((H, W), dtype=torch.bool)
            for _ in range(NUM_DOORS):
                door_span |= (self.grid_y >= (DOOR_POSITION - DOOR_SIZE)) & (
                    self.grid_y <= (DOOR_POSITION + DOOR_SIZE))
        else:
            wall_stripe = (self.grid_y >= (WALL_CENTER - half)) & (
                self.grid_y <= (WALL_CENTER + half))
            door_span = torch.zeros((H, W), dtype=torch.bool)
            for _ in range(NUM_DOORS):
                door_span |= (self.grid_x >= (DOOR_POSITION - DOOR_SIZE)) & (
                    self.grid_x <= (DOOR_POSITION + DOOR_SIZE))

        door_mask = wall_stripe & door_span
        wall_mask = wall_stripe & (~door_span)

        bs = BORDER_SIZE
        t = 4
        wall_mask[:, bs - t:bs] = True
        wall_mask[:, W - bs:W - bs + t] = True
        wall_mask[bs - t:bs, :] = True
        wall_mask[H - bs:H - bs + t, :] = True
        return wall_mask, door_mask

    def _render_base(self):
        """Background + doors + walls; the agent is blended on top per frame."""
        img = torch.empty((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
        for c in range(3):
            img[c].fill_(int(COLOR_BG[c]))
        if self._door_mask.any():
            for c in range(3):
                img[c, self._door_mask] = int(COLOR_DOOR[c])
        if self._wall_mask.any():
            for c in range(3):
                img[c, self._wall_mask] = int(COLOR_WALL[c])
        return img

    def _render_agent(self, agent_pos):
        """(H, W, C) uint8 -- the format dino_wm's Preprocessor expects."""
        alpha = self._gaussian_dot(agent_pos, AGENT_RADIUS).clamp(0, 1)
        out = self._base_img.to(torch.float32)
        for c in range(3):
            out[c] = out[c] * (1.0 - alpha) + float(COLOR_AGENT[c]) * alpha
        img = out.to(torch.uint8)
        return img.permute(1, 2, 0).numpy()

    def _apply_collisions(self, pos1, pos2):
        bs = float(BORDER_SIZE)
        door_margin = 1.75
        agent_r = AGENT_RADIUS

        x2, y2 = float(pos2[0]), float(pos2[1])
        x2 = min(max(x2, bs + agent_r), IMG_SIZE - bs - agent_r)
        y2 = min(max(y2, bs + agent_r), IMG_SIZE - bs - agent_r)
        pos2c = torch.tensor([x2, y2], dtype=torch.float32)

        half = WALL_THICKNESS // 2
        c = float(WALL_CENTER)
        if WALL_AXIS == 1:
            effective_left = c - half - agent_r
            effective_right = c + half + agent_r
            x1, x2_val = float(pos1[0]), float(pos2c[0])
            y2_val = float(pos2c[1])
            if x1 < c:
                if x2_val > effective_left and not self._in_any_door_1d(
                        y2_val, door_margin):
                    pos2c[0] = effective_left - 0.5
            else:
                if x2_val < effective_right and not self._in_any_door_1d(
                        y2_val, door_margin):
                    pos2c[0] = effective_right + 0.5
        else:
            effective_top = c - half - agent_r
            effective_bottom = c + half + agent_r
            y1, y2_val = float(pos1[1]), float(pos2c[1])
            x2_val = float(pos2c[0])
            if y1 < c:
                if y2_val > effective_top and not self._in_any_door_1d(
                        x2_val, door_margin):
                    pos2c[1] = effective_top - 0.5
            else:
                if y2_val < effective_bottom and not self._in_any_door_1d(
                        x2_val, door_margin):
                    pos2c[1] = effective_bottom + 0.5
        return pos2c

    @staticmethod
    def _in_any_door_1d(coord_1d, margin):
        for _ in range(NUM_DOORS):
            if (DOOR_POSITION - DOOR_SIZE - margin) <= coord_1d <= (
                    DOOR_POSITION + DOOR_SIZE + margin):
                return True
        return False
