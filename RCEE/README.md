
=== Question Generation

1. Clone https://github.com/facebookresearch/XLM to the project.

2. Follow `QuestionGeneration/data_preparation.py` to prepare the training data. In the directory, `train_source_example_inputs.txt` gives samples in the source side, i.e., the narrative-style texts; `train_target_what_questions.txt` and `train_target_when_questions.txt` show samples in the target side, i.e., the query-style texts.

3. Train back-translation model to learn the mapping. `generated_questions.txt` gives the examples of the generated questions.


=== Question Answering

1. Follow `util.py` and `model_train_progressly.py` to preprocess the datasets and train the final model.