from azureml.core import Workspace, Experiment

# ws = Workspace.get(name="udacity-project")
ws = Workspace.from_config()
exp = Experiment(workspace=ws, name="quick-starts-ws-137176")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()

----

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

cluster_name = 'project1-cluster'
try:
    cluster = ComputeTarget(workspace = ws, name = cluster_name)
    print("Cluster already exists, start using it")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", max_nodes = 4)
    cluster = ComputeTarget.create(ws, cluster_name, compute_config)

cluster.wait_for_completion(show_output = True)

--


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import choice, uniform

import os

# Specify parameter sampler
ps = RandomParameterSampling( {
    "--C": uniform(0.0, 1.0),
    "--max_iter": choice(10, 25, 50, 75, 100)
    }
)

# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval = 1, delay_evaluation = 5)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
pip_packages = []
est = SKLearn(
	source_directory='.',
	compute_target=cluster,
	entry_script = "train.py",
	pip_packages=pip_packages
)

# from azureml.core import ScriptRunConfig

# compute_target = ws.compute_targets['<my-cluster-name>']
# src = ScriptRunConfig(source_directory='.',
#                      script='train_iris.py',
#                      arguments=['--kernel', 'linear', '--penalty', 1.0],#
#                      compute_target=compute_target,
#                      environment=sklearn_env)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(
	estimator=est,
	hyperparameter_sampling=ps,
	policy=policy,
    primary_metric_name="AUC_weighted",
    primary_metric_goal=PrimaryMetricGoal.MINIMIZE	
)

---
from azureml.widgets import RunDetails
# Submit your hyperdrive run to the experiment and show run details with the widget.

hyperdrive_run = experiment.submit(hyperdrive_config)
RunDetails(hyperdrive_run).show()
hyperdrive_run.wait_for_completion(show_output=False)

---

import joblib
import os 

# Get your best run and save the model from that run.
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['Arguments']
print("\nBest run metrics:", best_run_metrics)
print("\n Parameter values:", parameter_values)

best_model = GradientBoostingRegressor(max_depth = 2, learning_rate = 1).fit(x_train, y_train)
best_model_filename = "hyperdrive_best_model.pkl" 

output_folder='./outputs'
os.makedirs(output_folder, exist_ok=True)
joblib.dump(value=best_model, filename=os.path.join(output_folder, best_model_filename))

---

from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

data_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path)
----

from train import clean_data

# Use the clean_data function to clean your data.
x, y = clean_data(ds)

---

from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="AUC_weighted",
    training_data=ds,
    label_column_name="y",
    n_cross_validations=3)
	
-----

# Submit your automl run
# ws, Experiment already available 
experiment_name = 'project1-automl'
experiment = Experiment(ws, experiment_name)
run = experiment.submit(automl_config)

# RunDetails(run).show()

run.wait_for_completion(show_output=True)

----
# Retrieve and save your best automl model.

best_automl_run, best_automl_model = run.get_output()
run_details = best_automl_run.get_details()
print("RUN_DETAILS", run_details)

print("PROPERTIES", run_details['properties'])

model_details = {
    'RunID': [run_details['runId']],
    'Iteration': [run_details['properties']['iteration']],
    'Primary metric': [run_details['properties']['primary_metric']],
    'Score': [run_details['properties']['score']],
    'Algorithm': [best_model.steps[1][0]],
    'Hyperparameters': [best_model.steps[1][1]]
}

# model_details_df = pd.DataFrame(model_details,
                  columns = ['RunID','Iteration','Primary metric','Score','Algorithm','Hyperparameters'],
                  index=[run_details['properties']['model_name']])

# pd.options.display.max_colwidth = -1

# model_details_df

model_file_name = 'automl_best_model.pkl'
joblib.dump(value=best_automl_model, filename=os.path.join(output_folder, model_file_name))

=======================

# Reference materials:
# https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c
# https://towardsdatascience.com/top-10-model-evaluation-metrics-for-classification-ml-models-a0a0f1d51b9
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train
# https://towardsdatascience.com/hidden-tricks-for-running-automl-experiment-from-azure-machine-learning-sdk-915d4e3f840e
# https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/how-to-track-experiments
