# Snake AI

An AI trainer and player for Snake in Pygame using PyTorch Reinforcement Learning.

`Python version = 3.11`

## Training
When training press `F` to speed up the game and `J` to disable rendering (lower computation).

To start training:
```bash
python ./SnakeAI.py
```

To train an existing model the model path can't be the default one.

## Playback
In the last block of `SnakeAI.py` you can call either `train()` or `play()`.

Both functions take a `model_path` argument, though `train()` will default to `./models/snake.pt`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
