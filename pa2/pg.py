import numpy as np
import gym
from tqdm import tqdm

def include_bias(ob):
    ob_with_bias = np.append(np.array([1]), ob)
    return ob_with_bias
    

def get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    action = np.random.multivariate_normal(mean, cov=[[1,0],[0,1]])
    if np.linalg.norm(action) > 0.025:
            action = 0.025 * action / np.linalg.norm(action)
        
    return action		

def grad_log_policy(state, theta, action):
    # print(include_bias(state), theta, action)
    # print("   ")
    mean = theta.dot(include_bias(state)).reshape((1, action.shape[0]))
    grad = (np.dot((action - mean).T, include_bias(state).reshape((1, state.shape[0] + 1))))
    # print(grad) 
    # print("  ")
    return grad

def policy_grad_method(env_id, gamma, max_iter, batch_size, rate_theta, rate_w, is_rendered):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    elif env_id == 'vishamC':
        from rlpa2 import vishamC
        env = gym.make('vishamC-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' or 'vishamC' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    
    for itr in (range(max_iter)):

        print("Iteration - " + str(itr))
        n_samples = 0
        
        del_theta = np.zeros((action_dim, obs_dim + 1))
        avg_reward = []
        lenth = []

        while n_samples < batch_size:
            trajectory = []
            is_done = False
            curr_state = env.reset()
            avg_r = 0
            while not is_done:
                if is_rendered:
                    env.render()
                
                curr_action = get_action(theta, curr_state)
                # if(np.linalg.norm(curr_action) < 0.024):
                #     print(np.linalg.norm(curr_action))
                next_state, reward_t, is_done, _ = env.step(curr_action)
                # print(next_state, curr_action, curr_state,reward_t, is_done)
                trajectory.append([curr_state, curr_action, reward_t, next_state])
                avg_r += reward_t
                curr_state = next_state
            # print(n_samples, " ", avg_r, " ", len(trajectory))
            avg_r /= len(trajectory)
            avg_reward.append(avg_r)
            lenth.append(len(trajectory))
            n_samples += 1
            grad_theta = np.zeros((action_dim, obs_dim + 1))
            G = 0
            
            for curr_seq in reversed(trajectory):
                
                S_t, A_t, r_t, _ = curr_seq

                G = r_t + (gamma * G)

                grad_theta += (G - avg_r)* grad_log_policy(S_t, theta, A_t)
                

            grad_theta /= (np.linalg.norm(grad_theta) + 1e-8)
            # print(grad_theta[1][0])
            del_theta += grad_theta
        
        theta += rate_theta * del_theta
        
        print("iteration - " , itr, "  Average Reward - ", (np.mean(avg_reward)), "  Theta - ", theta)

    return theta

print(policy_grad_method("chakra", 0.1, 1000, 20, 0.005, 0, False))