# 42_Learn2Slither
An Ai that plays Snake 🐍 🍎

Setup & usage:
    pip install -r requirements.txt
    ./src/main.py <ARGS>

# Usage
Here are the main arguments you might want to use:

1. Train your models, they will save to default location /save/the_model
    --train epoch_count 
        You can add optional flag to edit hyper parameters
        --epsilon [float: 0 < ARG < 1] 
        --decay [float] 
        --lr [float] 
        --batch_size [int]
   
3. You can test the model this way:
    --test number_of_sim_to_run --load_model <model_path>
    
3. You can Record a model-sim and the visualize it in a GUI
        --record --load_model <model_path>
        --visualize <record_path>
    
