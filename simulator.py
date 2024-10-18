import numpy as np
from single_process_mdp import SingleProcessMDP


NUM_RUNS = 25
NUM_STATES = 5

def select_action(env):
    return np.random.choice(env.na, p=[0.5, 0.5])

def log_history(time, old_state, action, curr_state, reward):
    print(f'{time+1}: OldS={old_state}, Action={action}, NewS={curr_state}, {reward}')


if __name__ == '__main__':
    env = SingleProcessMDP(NUM_STATES)
    env.reset()

    for t in range(NUM_RUNS):
        action = select_action(env)
        old_state = env.s
        curr_state, reward, job_complete = env.step(action)
        log_history(t, old_state, action, curr_state, reward)

        if job_complete:
            print("The job is complete")
            break
