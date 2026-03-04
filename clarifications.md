# Resources

- Models will be tested on CPU and will have limitations on test runtime and model checkpoint to model the limitations in the model scenario (i.e. astronauts using this model in space). See the README for more details.

- Your uncompressed dropbox submission is limited to 1 GB, not just the checkpoint. You don't need to include the `data` directory in your submission for this reason. See the README for details on what to include.

- The inference size is no more than 10k (for dropbox submission).

# Data

- There are no guarantees on the quality of the data e.g. there could be grammatical errors, typos, possible speech-to-text issues.

- There are no guarantees on which characters will be in the output vocabulary. You can assume it will be a valid unicode character. See `data/open-dev/answer.txt` for reference.

- The set of languages represented in `data/open-dev` is representative of the set of languages we'll be tested on (but it is not necessarily guaranteed that other languages won't appear in the test data). This is to emulate a real-world NLP project where the data provided by the user is their best-faith attempt to show you what real data looks like, but there are no guarantees.

- The test data may contain lines that have a combination of languages. In `/data/open-dev`, `lang.txt` will contain the predominant language of the speaker. For the purposes of this project, you can assume 1 language per line.

- Because your domain is narrow, you do not need internet-scale data. You need relevant data. You are looking for density of technical language, not breadth of general knowledge.

- `open-dev` is a disjoint split from `closed-dev` and `closed-test`.

- Since marking will not be case sensitive, you may choose to preprocess all training data and postprocess all predictions by lowercasing (when applicable to the language).

# Model

- You may use a pretrained model and further train it on a dataset we select, or implement a model from scratch, so long as it meets the resource constraints in the README.

- You are not training a general-purpose LLM. This is impossible given our time and compute constraints. Instead your goal is to build a highly specific, narrow-scope tool: an autocomplete system for astronauts.

- You should first establish a performance baseline using a simple statistical model, such as an N-gram model. This is to get past the hurdle of data gathering and pre-processing by first having a solid N-gram implementation with a clean data pipeline. This is a successful mid-term check=in.

- You may later swap the N-gram model for a transformer or fine-tune/distill an existing smaller model.

- With regards to using test input to compute aggregate statistics without updating weights (e.g. estimating character-level distributions) before generating predictions: Our inference jobs are ran exactly according to the README, so if we can somehow collect statistics under that setting then we're welcome to use it (note: this is likely not necessary).

# Evaluation

- The baseline naive implementation will be a small n-gram language model implemented by the teaching staff.

- For the Kaggle submission, our predictions should be for `test.csv`, which has about 59k rows. This is different from our zip dropbox submission using `input.txt` which has about 99k rows.

# Other

- You may enumerate the set of all characters and forgo a tokenizer, that is a design choice you may make.

- You can implement multiprocessing or other optimizations to help speed up the inference time, but it is not required. 

- The language zh is simplified mainland Chinese.