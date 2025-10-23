# Learn2Slither

An AI that plays Snake 🐍 🍎

#### Allowed libraries:

- Any helpers, as long as the core algorithms are still made by the student. For pedagogical reasons.

- I personally used torch to train a neural-net, and leverage batching operations.

### Features:

- A Gym-Simulator-Policy architecture: Where the `Gym` act as an orchestrator for any `Policy` to run in a `Simulator`
- Maurice Q-table (Good Student)
- Kevin Deep Q-Learning linear-neural-net (A cnn could have been better) (Bad student)
- A visualizer (pygame) to demonstrate runs
- An attempt at distributed training

# Usage

Setup & usage:
```sh
pip install -r requirements.txt
python3 ./src/main.py <ARGS>
```

Here are the main arguments you might want to use:

1. Train your models, they will save to default location /save/the_model
    - `--train <int: epoch_count> `.
You can add optional flag to edit hyper parameters
    - `--epsilon [float: 0 < ARG < 1] `
    - `--decay [float]`
    - `--lr [float] `
    - `--batch_size [int]`
You can also start from a pretrained model:
    - `--load_model <model_path>`

2. You can test the model this way:
    - `--test <int: number_of_sim_to_run> --load_model <model_path>`

3. You can Record a model simulation and then visualize it in a GUI
    - `--record --load_model <model_path>`
    - `--visualize <record_path>`

### Technical outline

Core constraint: Your snake can only see in the 4 directions from its head (North South East West as straight lines until a wall)

Used Python for pygame visualize, and easy libraries (torch). I wanted to focus on the theory more than the implementation.

### Skills & Lesson learned:

- Math fundamentals: Bellman equation, entropy & trivial basics.
- Distributed Training followed by Model Averaging / Ensemble Methods.
- Entropy and bias management.
- Soft skill: patience & compliance: I had to use flake8 which I don't like personally. I also had to face the limitation of the subject and couldn't implement the neural-net as wished.

Practical insight: pure-maths maps very badly to reality, in this instance it was about entropy management. If the samples where taken in chronological-order they would not generalize well enough, and if taken at random they would have been too slow to converge; Filtering the important ones against noise and sampling efficiently is a crucial case-specific problem.

### Notes:

My hypothesis on why Keving the Neural-net underperformed:
- The input space is very limited.
- memory/multi-frame is not allowed (Cannot have foresight/planning)
- I used only a stack of linear layers
- I had no idea what I was doing, I mostly focused on the Q-table by the end, such that I had something to submit (the neural-net was an optional requirement of the subject).
- I decided to learn more about neural-nets with a dedicated project right after.

My Q-table and Neural-nets are respectively named Maurice and Kevin.

It helps keeping yourself sane after hours of work: Thinking to yourself *"Kevin is just trolling"* lighten-up your mood as compared to *"Why won't my code work for f-sake"*.

**Some great resources I used to learn about Q-Learning:**
- *https://davidstarsilver.wordpress.com/teaching/* : Some excellent free lectures
