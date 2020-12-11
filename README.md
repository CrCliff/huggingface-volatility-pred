# huggingface-volatility-pred

Chris Clifford

To start:
```bash
$ pip install -r requirements.txt
$ jupyter lab
```

Then run all frames in `gen_data.ipynb`.
After that, run each frame in `train_test_split.ipynb`, then change the `DATA_FILE` variable to match what is in the `data/` directory.
Keep doing this until all sub-datasets have been generated (there should be 9 CSV files in `data/split/`.

Finally, run `./run.sh`. This will train and evaluate the model on whichever dataset the `BASE_FILE` variable matches.

Feel free to email `crc339@drexel.edu` with questions for running.