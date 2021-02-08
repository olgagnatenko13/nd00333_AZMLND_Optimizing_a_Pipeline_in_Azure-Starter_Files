path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path)

x, y = clean_data(ds)

# TODO: Split data into train and test sets.
test_size = 0.25
random_state = 7

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
