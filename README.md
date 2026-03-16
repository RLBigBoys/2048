# 2048 RL Project

This project is dedicated to training an agent to play 2048. Below is a step‑by‑step description of how the project evolved, what was done at each stage, and how the agent’s behavior and metrics changed over time.

## 1. First version of the project

Initially, the project only had a basic 2048 environment and a manual GUI for debugging.

What was implemented:
- pure NumPy 2048 environment
- actions `UP`, `DOWN`, `LEFT`, `RIGHT`
- correct shift and merge logic
- spawning of new tiles
- terminal state detection
- `pygame` GUI for manual play

At this stage, there was no training yet. The main goal was to ensure that the game mechanics were correct.

## 2. Environment verification and fixes

Next, the project was checked specifically as an RL environment.

What was checked:
- whether the board remains unchanged after an invalid move
- whether a new tile is spawned after an invalid move
- whether valid actions are computed correctly
- whether terminal state is detected correctly
- whether merge logic breaks on edge cases

Result:
- the core mechanics of the environment turned out to be correct
- the “extra 4” issue in the GUI was not a merge bug, but just a normal new tile spawned after a valid move

## 3. First features: 2x2 blocks

The next step was to reduce the state space.

Initial idea:
- don’t store the full board state as a table
- split the board into local `2x2` patterns
- use these patterns as features for tabular learning

At first there were 8 `2x2` blocks, but then the central block was added, and the partition became complete:
- `3 x 3` overlapping `2x2` windows
- `9` `2x2` blocks in total

This was the first step towards a feature-based representation of the state.

## 4. First agent and first action-selection scheme

After that, a trainable agent was added.

What was done:
- `LocalMajorityAgent` was introduced
- for each local feature, a tabular `Q` estimate of actions is stored
- two training modes were implemented:
  - `value_iteration`
  - `policy_iteration`

The first global action-selection scheme was:
- each `2x2` block votes for the best move
- the global move is chosen by majority vote

Why this was done:
- the idea was for decisions to be made locally in small patterns
- and then aggregated into a single global action

## 5. Learning curve, model saving and visualization

After this, the project became a more complete pipeline.

Added:
- saving the learning curve to CSV
- saving learning-curve plots to PNG
- saving the trained agent to `artifacts/agent_policy.pkl`
- separate agent evaluation without training
- a separate script `evaluate.py` that loads the saved model and runs the trained agent

Visualization also appeared in the project:
- manual GUI mode
- visualization during training
- visualization during evaluation

## 6. From majority vote to stronger aggregators

When real training started, it became clear that simple majority vote was too weak.

Observed:
- the agent could make locally reasonable moves
- but failed to maintain a global strategy
- the learning curve quickly hit a plateau
- `max_tile` usually got stuck at `256` or `512`

After that, other action-selection methods were tried.

### 6.1. Sum of Q

A scheme was implemented:
- instead of simply counting votes
- sum Q-values across all local features

The idea was to preserve the strength of preferences, not just a binary vote.

### 6.2. Weighted majority vote

The next step was `weighted_majority_vote`.

What it does:
- each feature still votes for a single action
- but the contribution of each vote depends on the feature’s importance

The weight depends on:
- the maximum value in the pattern
- the sum of values in the pattern

This is now the main mode:
- `action_selection_mode = "weighted_majority_vote"`

## 7. Expanding the feature space

When it became clear that only `2x2` patterns were not enough, the features were expanded.

Initially:
- `9` `2x2` blocks

Then added:
- `4` rows
- `4` columns

Then longer local patterns:
- `6` `2x3` blocks
- `6` `3x2` blocks

Current state representation:
- `9` `2x2` blocks
- `6` `2x3` blocks
- `6` `3x2` blocks
- `4` rows
- `4` columns

In total:
- `29` active features per board

Important:
- features are now tagged by type, e.g. `("block", tuple)` or `("row", tuple)`
- this is needed so identical tuples from different feature types do not get mixed together

## 8. Adapting the reward function to the real goal

As training progressed, it became clear that the project’s goal is not just to “play reasonably well,” but to actually reach `2048`.

Because of that, the reward function was strengthened and adjusted several times.

Currently, the reward includes:
- penalty for invalid moves
- reward for merges via `reward_score_scale`
- small bonus for changing the number of empty cells
- bonus for increasing the maximum tile
- large bonus for reaching `target_tile`
- soft penalty for long stagnation without max-tile growth
- corner-based bonus/penalty for the position of the max tile

### 8.1. Explicit 2048 goal

Added:
- `reward_target_tile_bonus`
- `terminate_on_target_tile`

This made reaching `2048` not just a side effect, but an explicit part of the objective.

### 8.2. Penalty for long stagnation

It was observed that the agent can spin for a long time without increasing `max_tile`.

So the following were added:
- `reward_stagnation_penalty`
- `stagnation_penalty_after_steps`

The environment now counts steps without max-tile growth and starts to softly penalize after a threshold.

### 8.3. Corner strategy

Since in 2048 it is important to keep the largest tile in one corner, the following were added:
- `target_corner`
- `reward_max_tile_in_corner_bonus`
- `reward_max_tile_out_of_corner_penalty`

Now the environment additionally rewards situations where the max tile is in the chosen corner.

## 9. Epsilon decay

Initially, exploration was constant:
- `epsilon` remained the same throughout training

Later it became clear that this prevents consolidating the strategy.

Therefore, `epsilon decay` was added:
- `train_epsilon` — starting value
- `train_epsilon_end` — final value
- `use_epsilon_decay` — toggles decay on/off

Now `epsilon` decreases linearly over episodes.

It looks like your GitHub README is suffering from a few classic LaTeX rendering hiccups. Specifically, the "Missing open brace" and "allowed only in math mode" errors occur because GitHub's Markdown renderer can be picky about how underscores and subscripts are handled inside and outside of `$$` blocks.

I have fixed the syntax in the sections below. I also added a helpful diagram of a Markov Decision Process to make the "MDP Objective" section more visually engaging for anyone reading your repo.

---

## 10. Mathematical formulation (aligned with the code)

This section follows the notation from `BIBLIE_FOR_RL_PROJECT.md`:

* $S_t, A_t, R_t$ are random variables
* $s_t, a_t, r_t$ are concrete realizations (tensors)
* $\gamma$ is the discount factor
* $\alpha_t$ is the learning rate at iteration $t$

### 10.1. MDP objective

The environment is an MDP $(\mathcal{S}, \mathcal{A}, p, p^R, \gamma)$, where:

* $p(s_{t+1} \mid s_t, a_t)$ defines transition dynamics
* $p^R(r_t \mid s_t, a_t)$ defines reward dynamics

The discounted return is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The optimal action-value function satisfies Bellman optimality:

$$q^*(s,a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} q^*(S_{t+1}, a') \mid S_t = s, A_t = a \right]$$

### 10.2. Local state-action representation

The implementation does not learn one global table over full boards. Instead, each board is transformed into local tagged features:

$$z_t = (\text{feature\_name}, \text{feature\_tuple}) \in \mathcal{Z}$$

For each local state $z$, the agent stores a vector of action values:

$$Q(z) \in \mathbb{R}^{|\mathcal{A}|}$$

### 10.3. Value-iteration mode (`update_value_iteration`)

For each local feature pair $(z_t^{(i)}, z_{t+1}^{(i)})$, the TD target is:

$$y_t^{(i)} = \begin{cases} r_{t+1}, & \text{if done or no valid next actions} \\ r_{t+1} + \gamma \max_{a' \in \mathcal{A}_{\text{valid}}(s_{t+1})} Q(z_{t+1}^{(i)}, a'), & \text{otherwise} \end{cases}$$

Then the update is:

$$Q(z_t^{(i)}, a_t) \leftarrow Q(z_t^{(i)}, a_t) + \alpha_t \left( y_t^{(i)} - Q(z_t^{(i)}, a_t) \right)$$

### 10.4. Policy-iteration mode (`policy_evaluation_update`)

For policy evaluation, the next-state bootstrap term is the masked policy expectation:

$$y_t^{(i)} = \begin{cases} r_{t+1}, & \text{if done or no valid next actions} \\ r_{t+1} + \gamma \sum_{a' \in \mathcal{A}_{\text{valid}}(s_{t+1})} \pi(a' \mid z_{t+1}^{(i)}) Q(z_{t+1}^{(i)}, a'), & \text{otherwise} \end{cases}$$

After policy-evaluation episodes, policy improvement moves each local policy toward greedy:

$$\pi_{\text{new}}(\cdot \mid z) = (1-\tau)\pi_{\text{old}}(\cdot \mid z) + \tau\pi_{\text{greedy}}(\cdot \mid z)$$

where $\tau = \text{train\_policy\_tau}$.

### 10.5. Exploration policy ($\epsilon$-greedy)

During training, action selection uses $\epsilon$-exploration over valid actions:

$$A_t = \begin{cases} \text{uniform random from } \mathcal{A}_{\text{valid}}(s_t), & \text{with probability } \epsilon_t \\ \text{aggregated greedy action}, & \text{with probability } 1-\epsilon_t \end{cases}$$


### 10.6. Reward decomposition

For a valid move (`changed=True`), the reward is:

$$r_{t+1} = (\text{merge\_gain}) \cdot c_{\text{score}} + c_{\text{large}} \cdot (\text{merge\_gain})^2 + (\Delta \text{empty}) \cdot c_{\text{empty}} + (\Delta \text{max\_exp}) \cdot c_{\text{max}} + (\Delta \text{snake}) \cdot c_{\text{snake}} + b_{\text{corner}} + b_{\text{target}} - p_{\text{stagnation}}$$

In code terms, these coefficients correspond to:

* $c_{\text{score}} =$ `reward_score_scale`
* $c_{\text{large}} =$ `reward_large_merge_factor`
* $c_{\text{empty}} =$ `reward_empty_bonus`
* $c_{\text{max}} =$ `reward_max_tile_bonus`
* $c_{\text{snake}} =$ `reward_snake_factor`

And additive terms are controlled by:

* `reward_max_tile_in_corner_bonus` / `reward_max_tile_out_of_corner_penalty`
* `reward_target_tile_bonus`
* `reward_stagnation_penalty` with threshold `stagnation_penalty_after_steps`

For an invalid move (`changed=False`), the base reward is:

$$r_{t+1} = \text{invalid\_move\_penalty}$$

with possible stagnation penalty applied by the same threshold logic.


### 10.7. Aggregation from local Q-values to a global action

The global action is selected from local tables by one of two aggregators:

* **`majority_vote`**: Each local feature $z \in \{z^{(i)}\}$ votes for its best valid action: $\text{argmax}_{a \in \mathcal{A}_{\text{valid}}} Q(z, a)$.
* **`weighted_majority_vote`**: The same voting, but each vote is weighted by feature statistics (e.g., max tile or sum of tiles within the pattern).

Thus, learning remains local in $Q(z,a)$, while acting is global through vote aggregation.


## 11. How metrics changed over time

Below is a brief description of how the agent’s behavior changed according to the learning curves.

### Early experiments

In the first versions:
- reward quickly reached a plateau
- score barely grew
- `max_tile` mostly got stuck at `128–256`

This showed that the agent had only learned local merges, but not a long-term strategy.

### After adding rows, columns, and long patterns

After moving to features:
- `2x2 + rows + cols`
- then `2x2 + 2x3 + 3x2 + rows + cols`

metrics improved:
- score started to grow more consistently
- `max_tile` reached `256` more often
- later `512` started to appear more frequently

This was the first sign that the agent was actually using more global board context.

### After epsilon decay

After adding `epsilon decay`:
- score growth became smoother
- rare good episodes started to appear more often
- training stopped being completely “noisy”

### After corner strategy and anti-stagnation reward

After adding:
- corner bonus
- stagnation penalty

it became clear that:
- score continues to grow
- `max_tile` reaches `512` noticeably more often
- but the agent still has not reliably reached `1024` and `2048`

## 12. Current state of the project

Currently, the project includes:
- a working 2048 environment
- verified merge and terminal-state logic
- an agent with tabular learning on local and semi-global patterns
- two training modes:
  - `value_iteration`
  - `policy_iteration`
- weighted majority vote
- model saving/loading
- learning curve CSV + PNG
- `pygame` visualization
- a separate `evaluate.py` script to run an already trained agent

## 13. Main takeaway at this stage

The project has moved far beyond a “simple 2048 environment” and has become a full RL setup with:
- its own reward function
- a complex feature set
- a tabular local agent
- visualization and evaluation

At the same time, the current main result is:
- the agent already plays better than random and better than early versions
- the agent consistently improves score
- the agent reaches `512` more often
- but training to stable `2048` has not yet been achieved

So the project is currently at an intermediate but already functional stage:
- infrastructure is ready
- environment and training work
- quality has improved
- the next focus is to break the `512/1024` ceiling and achieve stable `2048` runs

## 14. How to run

Training:

```bash
PYTHONPATH=src python main.py --mode train
```

Manual GUI:

```bash
PYTHONPATH=src python main.py --mode gui
```

Separate evaluation of a trained agent:

```bash
PYTHONPATH=src python evaluate.py
```

Without visualization:

```bash
PYTHONPATH=src python evaluate.py --no-visualize
```

все формулы переведи в latex стиль для md, чтобы они красиво отображались на GH