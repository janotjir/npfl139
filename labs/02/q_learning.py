#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2425.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0., type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")

# 0.6, 0.5, 0.8, 5000
# nope: 0.6, 0.3, 0.5, 5000
# nope: 0.4, 0.4, 0.9, 7.5k, 200-0.8
# almost: 0.4, 0.4, 0.9, 5000, 100-0.8


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Variable creation and initialization
    Q = 100*np.ones((env.observation_space.n, env.action_space.n))
    epsilon = args.epsilon

    training = True
    ep_count = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Perform an action.
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                #action = np.argmax(Q[state])
                action = argmax_with_tolerance(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            Q[state, action] += args.alpha * (reward + float(not done) * args.gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
        
        ep_count += 1
        #if ep_count % 250 == 0: epsilon = 0.8 * epsilon
        epsilon = 0.8**(ep_count / 250) * args.epsilon
        #epsilon = max(0.05, args.epsilon * (0.9995 ** ep_count))
        #epsilon = max(0.01, args.epsilon * (1 - ep_count / args.episodes)) # too slow
        if ep_count % 500 == 0: print(epsilon)
        if ep_count > args.episodes: break

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            #action = np.argmax(Q[state])
            action = argmax_with_tolerance(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
