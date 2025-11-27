import numpy as np
import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", is_slippery=False)
#env = gym.make("Taxi-v3")
#env = gym.make("CliffWalking-v0")
#env = gym.make("NChain-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

num_episodes = 3000
max_steps = 100

learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.001

rewards_per_episode = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = Q[state, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        q_values_next = Q[next_state, :]
        max_q_next = np.max(q_values_next)
        best_next_actions = np.where(q_values_next == max_q_next)[0]
        best_next_action = np.random.choice(best_next_actions)

        td_target = reward + gamma * Q[next_state, best_next_action]
        td_delta = td_target - Q[state, action]
        Q[state, action] += learning_rate * td_delta

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(rewards_per_episode[-500:])
        print(f"Episode {episode + 1}/{num_episodes} - Avg Reward (last 500): {avg_reward:.3f} - Epsilon: {epsilon:.3f}")

print("Training is done\n")
print("Final Q-Table:")
print(Q)

num_test_episodes = 3
max_steps_test = 100

print("\nTesting the trained agent in", num_test_episodes, "episodes:")

for episode in range(num_test_episodes):
    state, info = env.reset()
    total_reward = 0
    print(f"\nTest Episode {episode + 1}")

    for step in range(max_steps_test):
        q_values = Q[state, :]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

        if done:
            if reward == 1:
                print(f"✅ Reached the goal! Steps taken: {step + 1}")
            else:
                print(f"❌ Failed to reach the goal!")
            break

    print("Reward in this test episode:", total_reward)

env_render = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

num_render_episodes = 5
max_steps_render = 100

print("\nStarting visual rendering...\n")

for ep in range(num_render_episodes):
    state, info = env_render.reset()
    done = False
    print(f"Render Episode {ep + 1}")

    time.sleep(0.5)

    for step in range(max_steps_render):
        q_values = Q[state, :]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated
        state = next_state

        time.sleep(0.3)

        if done:
            break

    print(f"End of Render Episode {ep + 1}")
    time.sleep(1)

print("\nRendering completed. Close the window manually.\n")

while True:
    pass

