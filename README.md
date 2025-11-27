
# 🧊 Q-Learning on FrozenLake (Gymnasium RL Playground)

This repository contains a simple but complete implementation of **tabular Q-Learning** using environments from **[Gymnasium](https://gymnasium.farama.org/)**.

The main example uses the classic **FrozenLake-v1** environment (non-slippery version), but you can easily switch to other environments like **Taxi-v3**, **CliffWalking-v0**, or **NChain-v0**.

---

## ✨ Features

- ✅ Tabular Q-Learning implemented from scratch (no RL libraries)
- 🎯 ε-greedy exploration with decay
- 🧠 Handles multiple environments (FrozenLake, Taxi, CliffWalking, NChain)
- 📈 Training with average reward logging
- 👀 Visual rendering of the trained agent in FrozenLake
- 🧮 Final Q-table printed after training

---

## 📦 Requirements

Make sure you have **Python 3.8+** installed, then install the dependencies:

```bash
pip install numpy gymnasium[all]
````

> 🔍 Note: `gymnasium` is the new fork of OpenAI Gym, so we use `gymnasium` instead of `gym`.

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. Run the training script:

```bash
python frozenlake_Qlearning.py
```

During training, the script will print the **average reward** for every 500 episodes and the **current ε (epsilon)** value.

At the end, it will:

* Print the **final Q-table**
* Run a few **test episodes** without exploration
* Start a **visual rendering** of the trained agent on FrozenLake 🧊

---

## 🧪 Switching Environments

At the top of the script, you can change the environment manually:

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
# env = gym.make("Taxi-v3")
# env = gym.make("CliffWalking-v0")
# env = gym.make("NChain-v0")
```

Just **uncomment** the environment you want to try and **comment out** the others.

> ⚠️ Some environments may behave differently and may need hyperparameter tuning for best results.

---

## ⚙️ Algorithm Details

The agent uses:

* **Q-Table** of size `(n_states, n_actions)`
* **Update rule** (Q-Learning form with best next action):

[
Q(s, a) \leftarrow Q(s, a) + \alpha \big( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big)
]

Where:

* `α` – learning rate (`learning_rate = 0.8`)
* `γ` – discount factor (`gamma = 0.95`)
* `ε` – exploration rate (starts at `1.0` and decays each episode)

---

## 🎮 Rendering the Trained Agent

After training, the script creates a new environment with:

```python
env_render = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
```

Then it runs several **render episodes** using the learned Q-table so you can **watch the agent move on the grid**.

> 📝 You may need to close the rendering window manually when you're done.

---

## 📊 Hyperparameters

You can tweak these variables in the script:

```python
num_episodes = 3000
max_steps = 100

learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.001
```

Try changing them and see how it affects learning performance 🚀

---

## 🧑‍💻 Author

Created by Erfan Ramezani
