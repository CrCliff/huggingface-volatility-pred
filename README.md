# huggingface-volatility-pred
![Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiUkhMc2FCeXNuWDJzR0ovaURNRmRFcW9XTEFsajRMVUJ1RGFaTVd3V3ZhckZKdmFKdXFUMkNZWXgyZUl1ZU1HZGtlTi9uY0VLb0pnb1FjdEloanlmeEpVPSIsIml2UGFyYW1ldGVyU3BlYyI6IjJXMUVHTitMSEVCWXFJN0ciLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=main)

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
