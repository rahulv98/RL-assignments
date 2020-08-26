import numpy as np
import os

from simple_dqn_cartpole_skeleton import DQN

MINI_BATCH_SIZES = [16, 32, 64, 128]
LEARNING_RATES = [0.01, 0.001, 0.0001]
EPSILON_STARTS = [0.5, 0.7, 1]
REPLAY_MEMORY_SIZES = [1000, 10000, 100000]
SEEDS = [1, 345, 50, 278]

def exp_mini_batch_size(sizes, seed, env="CartPole-v0"):
    os.makedirs("results/exp_mini_batch_size/", exist_ok=True)
    for run, size in enumerate(sizes):
        dqn = DQN(env, seed, plot=False)
        dqn.MINIBATCH_SIZE = size
        dqn.initialize_network()
        
        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/exp_mini_batch_size/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_learning_rate(rates, seed, env="CartPole-v0"):
    os.makedirs("results/exp_learning_rate/", exist_ok=True)
    for run, rate in enumerate(rates):
        dqn = DQN(env, seed, plot=False)

        dqn.LEARNING_RATE = rate
        dqn.initialize_network()
        
        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/exp_learning_rate/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_start_epsilion(eps, seed, env="CartPole-v0"):
    os.makedirs("results/exp_start_epsilion/", exist_ok=True)
 
    for run, ep in enumerate(eps):
        dqn = DQN(env, seed, plot=False)

        dqn.EPSILON_START = ep
        dqn.initialize_network()
        
        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/exp_start_epsilion/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_replay_memory(replay_memory_sizes, seed, env="CartPole-v0"):
    os.makedirs("results/exp_replay_memory_size/", exist_ok=True)
    for run, size in enumerate(replay_memory_sizes):
        dqn = DQN(env, seed, plot=False)

        dqn.REPLAY_MEMORY_SIZE = size
        dqn.initialize_network()
        
        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/exp_replay_memory_size/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_without_replay(seeds, env="CartPole-v0"):
    os.makedirs("results/learning_curve_without_replay", exist_ok=True)
    
    for run, seed in enumerate(seeds):
        dqn = DQN(env, seed, plot=False, replay_req=False)
        dqn.initialize_network()

        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/learning_curve_without_replay/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_without_target(seeds, env="CartPole-v0"):
    os.makedirs("results/learning_curve_without_target", exist_ok=True)
    
    for run, seed in enumerate(seeds):
        dqn = DQN(env, seed, plot=False, target_req=False)
        dqn.initialize_network()

        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/learning_curve_without_target/run_" + str(run) + ".npy", dqn.ep_rewards)

def exp_without_replay_and_target(seeds, env="CartPole-v0"):
    os.makedirs("results/learning_curve_without_replay_and_target", exist_ok=True)
    
    for run, seed in enumerate(seeds):
        dqn = DQN(env, seed, plot=False, replay_req=False, target_req=False)
        dqn.initialize_network()

        print("\nStarting training...\n")
        dqn.train()
        print("\nFinished training...\n")

        np.save("results/learning_curve_without_replay_and_target/run_" + str(run) + ".npy", dqn.ep_rewards)


if __name__ == "__main__":
    print(" \n \n### experimenting without target network #####  \n \n")
    exp_without_target(SEEDS)
    print(" \n \n ### experimenting without replay memory #####  \n \n")
    exp_without_replay(SEEDS)
    print(" \n \n### experimenting without target network and replay memory #####  \n \n")
    exp_without_replay_and_target(SEEDS) 
    print("\n \n ### experimenting mini batch sizes #####  \n \n")
    exp_mini_batch_size(MINI_BATCH_SIZES, SEEDS[0])
    print(" \n \n ### experimenting replay memory sizes ##### \n \n")
    exp_replay_memory(REPLAY_MEMORY_SIZES, SEEDS[0])
    print(" \n \n ### experimenting epsilion start values #####  \n \n")
    exp_start_epsilion(EPSILON_STARTS, SEEDS[0])
    print(" \n \n ### experimenting learning rates ##### \n \n")
    exp_learning_rate(LEARNING_RATES, SEEDS[0])
    
    
    
    
    