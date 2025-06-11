import argparse, os, csv
import yaml
import numpy as np
from taxi_dqn.utils.environment import Environment
from taxi_dqn.utils.agent import DQNAgent, AgentConfig
from taxi_dqn.utils.graphs import plot_training, plot_test_hist
from tqdm import tqdm

# helper for configs :) not perfect but for this case is OK
def load_cfg(mode_cfg_path, agent_cfg_path):

    with open(mode_cfg_path) as f:
        mode_cfg = yaml.safe_load(f)

    with open(agent_cfg_path) as f:
        agent_cfg_yaml = yaml.safe_load(f)

    agent_cfg = AgentConfig(
        state_dim = agent_cfg_yaml["state_dim"],
        action_dim = agent_cfg_yaml["action_dim"],
        hidden_sizes = agent_cfg_yaml["hidden_sizes"],
        use_embedding = agent_cfg_yaml["use_embedding"],
        embedding_dim = agent_cfg_yaml["embedding_dim"],
        gamma = agent_cfg_yaml["gamma"],
        lr = agent_cfg_yaml["learning_rate"],
        batch_size = agent_cfg_yaml["batch_size"],
        buffer_size = agent_cfg_yaml["replay_memory_size"],

        epsilon = mode_cfg["epsilon_start"],
        epsilon_min = mode_cfg["epsilon_end"],
        epsilon_decay = mode_cfg["epsilon_decay_rate"],
        device = mode_cfg["device"],
    )
    return mode_cfg, agent_cfg

# train logic
def train(train_cfg, agent_cfg):

    print("LOADING AGENT AND ENV!")
    env = Environment.make_env(seed=train_cfg["seed"], render=train_cfg["render"])
    agent = DQNAgent(agent_cfg)
    rewards, avg100, epsilons, losses, successes = [], [], [], [], []

    # ensure output dirs exist or make them
    os.makedirs(train_cfg["log_csv"].split("/")[0], exist_ok=True) # results dir
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True) # ckpts dir

    # reset csv log for data dumping;
    with open(train_cfg["log_csv"], "w", newline="") as f:
        csv.writer(f).writerow(["episode","reward","epsilon","loss","success"])

    print("STARTING TRAINING!")
    print(20*"----")

    with open(train_cfg["log_csv"], "a", newline="") as fout:

        for ep in tqdm(range(1, train_cfg["episodes"] + 1)):
            env.render() # if render is set to true, pygame gui becomes visible
            state, _ = env.reset()
            done, ep_reward, ep_losses = False, 0, []

            # simulate until success or end
            while not done: 
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                loss_val = agent.update()

                if loss_val is not None:
                    ep_losses.append(loss_val)

                agent.push(state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward

            agent.decay_epsilon()
            if ep % train_cfg["target_update_interval"] == 0:
                agent.update_target()

            # store appropriate results
            rewards.append(ep_reward)
            avg100.append(np.mean(rewards[-100:]))
            epsilons.append(agent.epsilon)
            losses.append(np.mean(ep_losses) if ep_losses else np.nan)
            successes.append(1 if terminated else 0)

            # log episode info to csv
            csv.writer(fout).writerow([ep, ep_reward, agent.epsilon,
                                    losses[-1], successes[-1]])

            # print info to the user
            # TODO: switch to loggers;;
            if ep % 10 == 0:
                print(f"CURR. EPISODE: {ep:4}/{train_cfg['episodes']} \n"
                    f"EP. REWARD: {ep_reward:4} \nAVERAGE REWARD OVER LAST 100 EP.: {avg100[-1]:6.2f} \n"
                    f"AVERAGE ACC. OF SUCC. OVER LAST 100 EP.: {(sum(successes[-100:])/100):6.2f} \nEPSILON: {agent.epsilon:.3f}")
                print(20*"----")

            if ep % train_cfg["checkpoint_interval"] == 0:
                agent.save(f"{train_cfg['checkpoint_dir']}/dqn_taxi_ep_{ep}.pt")

    print("TRAINING FINISHED, SAVING MODEL!")
    # final save and plots
    final_path = f"{train_cfg['checkpoint_dir']}/dqn_taxi_final.pt"
    agent.save(final_path)
    print("Model saved:", final_path)

    plot_training(rewards, avg100, epsilons, losses, successes, train_cfg["plot_path"])

# test logic
def test(test_cfg, agent_cfg):
    env = Environment.make_env(seed=test_cfg["seed"], render=test_cfg["render"])
    agent = DQNAgent(agent_cfg)
    agent.load(test_cfg["model_checkpoint"])

    n_eps = test_cfg["episodes"]
    print(f"\n--- TESTING FOR {n_eps} EPISODES ---")

    rewards = []
    for ep in tqdm(range(1, n_eps + 1)):
        env.render()
        print(f"Episode {ep}/{n_eps}")

        state, _ = env.reset()
        done, ep_reward, steps = False, 0, 0

        while not done:
            action = agent.choose_action(state, exploit=True)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward
            steps += 1

        print(f"FINISHED IN {steps} STEPS | EPISODE REWARD: {ep_reward}\n")
        rewards.append(ep_reward)

    avg = np.mean(rewards)
    print(f"\n--- TEST COMPLETE — AVG. REWARD OVER {n_eps} EP.: {avg:.2f} ---")

    with open(test_cfg["log_csv"], "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["episode", "reward"])
        for ep, r in enumerate(rewards, 1):
            writer.writerow([ep, r])
    
    print("INFO LOGGED TO:", test_cfg["log_csv"])

    # plot! :)
    plot_test_hist(rewards, test_cfg["plot_path"])
    env.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--mode-cfg", default="configs/train.yaml")
    parser.add_argument("--agent-cfg", default="configs/agent.yaml")
    args = parser.parse_args()

    mode_cfg, agent_cfg = load_cfg(args.mode_cfg, args.agent_cfg)

    if args.mode == "train":
        train(mode_cfg, agent_cfg)
    else:
        # override agent's epsilon for testing!!
        agent_cfg.epsilon = mode_cfg["epsilon"]
        test(mode_cfg, agent_cfg)

if __name__ == "__main__":
    main()
