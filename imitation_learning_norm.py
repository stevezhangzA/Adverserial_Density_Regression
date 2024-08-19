from imitation_learning_cfg import *
from utils import *
from modules import *
from trainer import *
from tensorboardX import SummaryWriter

@pyrallis.wrap()
def train(config: TrainConfig):
    unnormalized_return = []

    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    max_steps = env._max_episode_steps
    ###################################
    #       load the datasets         #
    ###################################
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)                                 # sub-optimal datasets
    if not config.reward_sparse:
        expert_data = d4rl.qlearning_dataset(gym.make(config.expert_data))    # expert datasets:
        expert_data = filter_expert(expert_data,                              # sampled the topk expert trails as demonstration
                                    config.topk)
    else:
        expert_data= d4rl.qlearning_dataset(gym.make(config.expert_data))
        expert_data=filter_expert(expert_data,                                # if current dataset is reward sparse task
                                  config.topk)                                # then utilizing the topk trails as demonstration.
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)  # sub-optimal normalization
        state_mean_optim, state_std_optim = compute_mean_std(expert_data["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)  # sub-optimal normalization
    state_mean_optim, state_std_optim = compute_mean_std(expert_data["observations"], eps=1e-3)
    # normalizing sub-optimal dataset
    dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    # normalizing optimal dataset
    expert_data["observations"] = normalize_states(expert_data["observations"], state_mean_optim, state_std_optim)
    expert_data["next_observations"] = normalize_states(expert_data["next_observations"], state_mean_optim, state_std_optim)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    # data dim
    expert_dim=expert_data['rewards'].shape[0]
    non_expert_dim=dataset['rewards'].shape[0]
    expert_plus_nonexp_dim=expert_dim+non_expert_dim

    ###################################
    #       build replay buffer       #
    ###################################  

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        expert_plus_nonexp_dim,
        config.device,
    )
    replay_buffer_good = ReplayBuffer(
        state_dim,
        action_dim,
        expert_dim,
        config.device,
    )
    replay_buffer_bad = ReplayBuffer(
        state_dim,
        action_dim,
        non_expert_dim,
        config.device,
    )
    replay_buffer_good.load_d4rl_dataset(expert_data)
    replay_buffer_bad.load_d4rl_dataset(dataset)
    catted_data = {}
    for k in dataset:
        catted_data[k] = np.concatenate([dataset[k], expert_data[k]], axis=0)
    replay_buffer.load_d4rl_dataset(catted_data)

    max_action = float(env.action_space.high[0])
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)
    # initializing modules
    # vae good
    if not config.use_vqvae:
        vae_good = VAE(
            state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
        ).to(config.device)
        vae_optimizer_good = torch.optim.Adam(vae_good.parameters(), lr=config.vae_lr)
        # vae bad
        vae_bad = VAE(
            state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
        ).to(config.device)
        vae_optimizer_bad = torch.optim.Adam(vae_bad.parameters(), lr=config.vae_lr)
    else:
        vae_good = VQVAE(
            state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
        ).to(config.device)
        vae_optimizer_good = torch.optim.Adam(vae_good.parameters(), lr=config.vae_lr)
        # vae bad
        vae_bad = VQVAE(
            state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
        ).to(config.device)
        vae_optimizer_bad = torch.optim.Adam(vae_bad.parameters(), lr=config.vae_lr)
    actor = Actor(state_dim, action_dim, max_action, config.actor_init_w).to(
        config.device)
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)       
    # configuring modules
    kwargs = {"max_action": max_action,
              "vae_good": vae_good,
              "vae_optimizer_good": vae_optimizer_good,
              "vae_bad": vae_bad,
              "vae_optimizer_bad": vae_optimizer_bad,
              "discount": config.discount,
              "tau": config.tau,
              "device": config.device,
              # TD3
              "policy_noise": config.policy_noise * max_action,
              "noise_clip": config.noise_clip * max_action,
              "policy_freq": config.policy_freq,
              "actor": actor,
              "actor_optimizer": actor_optimizer,
              # SPOT
              "beta": config.beta,
              "lambd": config.lambd,
              "num_samples": config.num_samples,
              "iwae": config.iwae,
              "lambd_cool": config.lambd_cool,
              "lambd_end": config.lambd_end,
              "max_online_steps": config.online_iterations,
              "weighted_estimation":config.weighted_estimation}
    print("---------------------------------------")
    print(f"Training SPOT, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = adverseial_density(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    print("Training VAE good")
    for t in range(int(config.vae_iterations)):
        batch = replay_buffer_good.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.vae_good_train(batch)
        batch_suboptimal = replay_buffer_bad.sample(config.batch_size)
        trainer.dual_estimation_objective_optim(batch_suboptimal,batch,
                                                weight=1)
        log_dict["vae_iter"] = t
        print(log_dict)
        wandb.log(log_dict, step=trainer.total_it)

    print("Training VAE bad")
    for t in range(int(config.vae_iterations)):
        batch = replay_buffer_bad.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.vae_bad_train(batch)
        log_dict["vae_iter"] = t
        print(log_dict)
        wandb.log(log_dict, step=trainer.total_it)

    vae_good.eval()
    vae_bad.eval()

    state, done = env.reset(), False
    eval_successes = []
    train_successes = []
    evaluate_performance_record = []
    print("Offline pretraining")
    if not os.path.exists(config.log_saving_dir):
        os.mkdir(config.log_saving_dir)
    if not os.path.exists(os.path.join(config.log_saving_dir,config.project_name)):
        os.mkdir(os.path.join(config.log_saving_dir,config.project_name))
    writer = SummaryWriter(os.path.join(config.log_saving_dir,config.project_name,str(config.seed)))

    if 'kitchen' in config.env:
        rewards=dataset['rewards']
        total_stages=len(set(rewards))

    for t in range(int(config.offline_iterations)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.max_train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            if "kitchen" not in config.env:
                eval_scores, success_rate = eval_actor(eval_env,
                                                    actor,
                                                    device=config.device,
                                                    n_episodes=config.n_episodes,
                                                    seed=config.seed, )
            else:
                eval_scores, success_rate = eval_kitchen_mix(eval_env,
                                                            actor,
                                                            device=config.device,
                                                            n_episodes=config.n_episodes,
                                                            seed=config.seed,
                                                            total_stages= total_stages)
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            print(normalized)
            # print(f'score: {eval_scores}')
            # eval_scores.append(normalized)
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            unnormalized_return.append(normalized)
            normalized_eval_score = normalized * 100.0
            evaluate_performance_record.append(normalized_eval_score)
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            writer.add_scalars('data/scalar_group', eval_log  , t)
            #writer.add_scalars('data/scalar_group', normalized, t)
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(eval_log, step=trainer.total_it)
        with open(os.path.join(config.log_saving_dir,
                               config.project_name,
                               str(config.seed)+'.pkl'), 'wb') as f:
            pkl.dump([evaluate_performance_record, unnormalized_return], f)

if __name__ == "__main__":
    train()
