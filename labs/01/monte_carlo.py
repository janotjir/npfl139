#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=2500, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(args.episodes):
        # TODO: Perform an episode, collecting states, actions and rewards.
        episode = []

        state, done = env.reset()[0], False
        while not done:
            # TODO: Compute `action` using epsilon-greedy policy.
            if np.random.uniform() < args.epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(Q[state])

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode.append((state, action, reward))
            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + G
            C[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / C[state, action]

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
