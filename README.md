# traffic-control-system
AI course finel project

### Running Instructions
For running the environment:
```angular2
python3 test.py [-d (optional)] [-a <model>] [-c <config_file_path>] [-m <path_to_trained_DQN_model>] [-n <n_training_for_q_learning_agent> (optional)] [-p <path_to_weights_file> (optional)]
```
For training a DQN model:
```angular2
python3 DQNTrainer.py [-c <config_file_path>] [-p <path_to_save_the_trained_model>] [--n_actions <int>] [--n_episodes <int>] [--max_iterations <int>] [-b <batch_size>] [-g <gamma>]
```
