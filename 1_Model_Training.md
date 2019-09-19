
# Model Training
To set up a project on AML service, we are going to train it locally first and then upload it in Azure once it has been properly constructed.

In this example, we are going to use the [diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) dataset from the sklearn library. We are going to perform a simple Ridge Regression to predict the disease progression of the patients in the dataset.

In order to train a model in the cloud, we are going to follow these steps:
- __Train a model locally__ <br>
This allows to make sure your model is properly working and save Azure credits
- __Create a workspace in Azure__ <br>
Here is the place where all of our work in the cloud is going to be stored.
- __Create an experiment in the workspace__ <br>
An experiment is the place where we'll be executing our scripts, save and deploy our model.
- __Create a compute target__ <br>
Specifies the type of machine that is going to execute our code.
- __Import the data in the cloud__ <br>
The data used to train your model has to be stored in the cloud.
- __Create the training script__ <br>
This is the script that is going to be executed to train our model in the cloud.
- __Configure and execute the run__ <br>
The run will execute the training script in the cloud.
- __Regiser the model in the workspace__ <br>
The model traind in the run will be stored in the workspace so that it can later be used for deployment.

This section is mainly based on this article: https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml

### Train a model locally


```python
# Libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Data preprocessing
X, y = load_diabetes(return_X_y=True)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ridge Regression
alpha = 0.1
reg = Ridge(alpha=alpha)
reg.fit(X_train, y_train)

# Prediction and evaluation
preds = reg.predict(X_test)
mse = mean_squared_error(preds, y_test)
print("The Mean Squared Error on the test set is %d" %mse)
```

    The Mean Squared Error on the test set is 3372
    

### Create a workspace

Now we are going to upload the model in the cloud. The model will be stored in an experiment in a workspace in Azure Machine Learning service Workspace. A Worskpace is the place where all experiments, and thereby models, are going to be stored. It serves as a hub for building and deploying models.

A workspace can be manually created via the Azure portal or running the following cell.


```python
import azureml.core # azureml-sdk library

# Create the workspace
from azureml.core import Workspace

ws = Workspace.create(name = "create_a_workspace_name", # Workspace name, choose the name you like
                      subscription_id = "SET_YOURS", # Your Azure's subscription id
                      resource_group = "create_a_resource_group", # Resource group name, choose the name you like
                      create_resource_group = True,
                      location = "eastus2", # Place where your workspace will be located
                      exist_ok = True)
ws.get_details()
ws.write_config()
```

    WARNING - Warning: Falling back to use azure cli login credentials.
    If you run your code in unattended mode, i.e., where you can't give a user input, then we recommend to use ServicePrincipalAuthentication or MsiAuthentication.
    Please refer to aka.ms/aml-notebook-auth for different authentication mechanisms in azureml-sdk.
    

Wait some minutes until the workspace has been created. To visually check the workspace, go to the [Azure Portal](portal.azure.com) -> search "Machine Learning service workspaces" and click on the Workspace you just created.

Next step is to connect to the workspace.


```python
# Connect to the workspace
ws = Workspace.from_config()
```

### Create an experiment in the workspace

An experiment is a collection of runs. A run is an execution of Python code that does a specific task, such as training a model.

An experiment can as well be created viat the Azure portal or with the following cell.


```python
# Create an experiment in the workspace
from azureml.core import Experiment
exp = Experiment(workspace = ws, # Workspace to store our Experiment. Here, it's the previously created variable ws
                 name = 'create_an_experiment_name') # Experiment name, choose the name you like
```

You can take a look at your newly created Experiment in the Azure Portal. Go to Machine Learning service workspaces -> Click on your workspace -> Experiments.

### Create a compute target

A compute target is the compute resource to run a training script or to host a service deployment. It is attached to a workspace. It specifies the type of machine on which your experiment is going to be executed.


```python
# Create a compute target
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")
compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                min_nodes=compute_min_nodes,
                                                                max_nodes=compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())
```

    found compute target. just use it. cpucluster
    

### Upload the data in the cloud

Uploading the data into a blob storage in Azure will allow the training script to import the data. The data to be uploaded is the .csv file in the _data_ folder from this repository. This .csv file corresponds to _X_train_.


```python
import os
ds = ws.get_default_datastore()

# Import the data in a blob storage in the storage account
ds.upload(src_dir = os.path.join(os.getcwd(), 'data'), target_path = "data", overwrite = True)
```

    Uploading an estimated of 1 files
    Uploading C:\Users\a.nogue.sanchez\OneDrive - Avanade\Documents\Projects\Git - Machine Learning in Power BI using Azure Machine Learning service\data\diabetes.csv
    Uploaded C:\Users\a.nogue.sanchez\OneDrive - Avanade\Documents\Projects\Git - Machine Learning in Power BI using Azure Machine Learning service\data\diabetes.csv, 1 files out of an estimated total of 1
    Uploaded 1 files
    




    $AZUREML_DATAREFERENCE_ec981c52853746fb8594cd53b9c9830e



We can visually check if the data has properly been uploaded in the Azure Portal. Go to Storage Accounts -> Click on the storage account (which should have a name similar to your workspace) -> Click on Blobs -> Click on the newly created Blob -> data.

### Create the training script

This is the script that is going to be executed in the run. In our case, it will contain a Machine Learning model training. The training script must return a .pkl file containing the model. We decided to add some logs to the model, in this case an evaluation metric (MSE) and the hyperparameters used.


```python
%%writefile train.py

# Libraries
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from azureml.core import Run
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str)
args = parser.parse_args()
data_folder = args.data_folder
print('Data folder:', data_folder)

# Data preprocessing
X, y = load_diabetes(return_X_y=True)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

# get hold of the current run
run = Run.get_context()

# Ridge Regression
alpha = 0.1
reg = Ridge(alpha=alpha)
reg.fit(X_train, y_train)

# Prediction and evaluation
preds = reg.predict(X_test)
mse = mean_squared_error(preds, y_test)
run.log('alpha', alpha)
run.log('mse', mse)

os.makedirs('outputs', exist_ok=True)

# Save the model: reg
joblib.dump(value=reg, filename='outputs/diabetes_regression.pkl')
```

    Overwriting train.py
    

### Configure and execute the run

An SKLearn estimator object is used to submit the run. It allows to specify the instructions for executing the run.


```python
# Create an estimator
from azureml.train.sklearn import SKLearn

script_params = {
    '--data-folder': ds.path('data').as_mount()
}

est = SKLearn(source_directory=os.getcwd(),
              script_params=script_params,
              compute_target=compute_target,
              entry_script='train.py')
```

Submit the run. This might take around 10 minutes. To check the status of the run, click on "Link to Azure Portal" and wait until the Status "Completed".


```python
# Submit the job to the cluster
run = exp.submit(config=est)
run
```




<table style="width:100%"><tr><th>Experiment</th><th>Id</th><th>Type</th><th>Status</th><th>Details Page</th><th>Docs Page</th></tr><tr><td>create_an_experiment_name</td><td>create_an_experiment_name_1568707675_189337f3</td><td>azureml.scriptrun</td><td>Queued</td><td><a href="https://mlworkspace.azure.ai/portal/subscriptions/9e8d74ab-518e-4aa2-91eb-f1606e7312b6/resourceGroups/create_a_resource_group/providers/Microsoft.MachineLearningServices/workspaces/create_a_workspace_name/experiments/create_an_experiment_name/runs/create_an_experiment_name_1568707675_189337f3" target="_blank" rel="noopener">Link to Azure Portal</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.script_run.ScriptRun?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



Once the run has been completed, we can check whether the model has been correctly built. To do so, we can recover the logs of the run to see if they match with what we expect from the model. Here, we got alpha = 0.1 and  MSE = 3373, which is what the model should have returned.


```python
# Execute the cell only when the run has been completed
run.wait_for_completion(show_output=False)

# Get the run metrics
print(run.get_metrics())
```

    {'alpha': 0.1, 'mse': 3372.649627810032}
    

### Register the model
Once the run has been completed, we can register the model of that run in order to be able to use it later for deployment.


```python
# Register the model
model = run.register_model(model_name='diabetes_regression',
                           model_path='outputs/diabetes_regression.pkl')
print(model.name, model.id, model.version, sep='\t')
```

    diabetes_regression	diabetes_regression:1	1
    

The model is now correctly registered in the Workspace. See the next notebook __2_Model_Deployment__ to see how to deploy this model to later invoke it in Power BI.

Note that, to avoid further costs, we can now delete the compute target, as it was only requied for executing the run.


```python
compute_target.delete()
```
