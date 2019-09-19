
# Model deployment

This notebook explains how to deploy the model that we have trained and registered in an Azure workspace in the previous notebook. The model will be deployed as a web service in [Azure Container Instances](https://docs.microsoft.com/en-gb/azure/container-instances/).

A web service is an image, in this case a Docker image. It encapsulates the scoring logic and the model itself. This image is what will later be called in Power BI.

To build the correct environment for Container Instances, provide the following components:

- A scoring script to show how to use the model.
- An environment file to show what packages need to be installed.
- A configuration file to build the container instance.
- The model we trained in the previous notebook.

This section is mainly based on the following link: https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-deploy-models-with-aml.

### Retrieve the model from the Workspace

Before deploting the model, we should first download it from the Workspace.


```python
# Retrieve the model from the workspace
from azureml.core import Workspace
from azureml.core.model import Model
ws = Workspace.from_config()
model = Model(ws, 'diabetes_regression')

model.download(target_dir=os.getcwd(), exist_ok=True)

# verify the downloaded model file
file_path = os.path.join(os.getcwd(), "diabetes_regression.pkl")

os.stat(file_path)
```

    WARNING - Warning: Falling back to use azure cli login credentials.
    If you run your code in unattended mode, i.e., where you can't give a user input, then we recommend to use ServicePrincipalAuthentication or MsiAuthentication.
    Please refer to aka.ms/aml-notebook-auth for different authentication mechanisms in azureml-sdk.
    




    os.stat_result(st_mode=33206, st_ino=9851624185204729, st_dev=270394477, st_nlink=1, st_uid=0, st_gid=0, st_size=637, st_atime=1568707412, st_mtime=1568707948, st_ctime=1568619226)



### Create a scoring script

The scoring script contains the instructions to deploy the model as a web service. It must include the two following functions:

- ```python
init()
```  
Loads the model saved in the Workspace into a global object.
- ```
run(input_data)
```
Uses the model to make predictions based on the input data.

In addition, the scoring script must provide instructions to AML to generate a schema file. A schema file is what is going to tell Power BI what is expected as input and what is going to be returned by the model. In other words, the scoring script must provide a sample of the input and output of the deployed model. See this links to have futher information on schema generation [link1](https://docs.microsoft.com/en-us/power-bi/service-machine-learning-integration), [link2](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#example-script-with-dictionary-input-support-consumption-from-power-bi). This step is very important as Power BI won't be able to invoke a model without schema.

Note that schema generation is note required for model deployment but it is for invoking the deployed web service in Power BI. If you deploy a web service with AML Studio, the schema generation is automatic.


```python
%%writefile score.py

# Create a scoring script
import json
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.externals import joblib
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from azureml.core.model import Model

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('diabetes_regression')
    model = joblib.load(model_path)
    
# Schema generation
# Must be set in the scoring script, out of the init() and run() functions.
# In our case, we have a dataframe as input with 10 variables: 0 to 9 and they are all numeric.
input_sample = pd.DataFrame(data=[{
    "0": 0.1,
    "1": 0.1,
    "2": 0.1,
    "3": 0.1,
    "4": 0.1,
    "5": 0.1,
    "6": 0.1,
    "7": 0.1,
    "8": 0.1,
    "9": 0.1
}])

# As our model is a regression, the output is numeric and, in our case, the model returns an array.
# Check the type of your input and output to specify the correct format of your input and output sample.
output_sample = np.array([241.1])

# Specify the type of your input and output
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample)) 

def run(data):
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
```

    Overwriting score.py
    

### Environment file

We create an environment file, called _myenv.yml_, that specifies all of the script's package dependencies. This file is used to make sure that all of those dependencies are installed in the Docker image.

This file can be created via Python code (follow the instrctions [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-deploy-models-with-aml)) but as I particularly struggled to include the __inference_schema__ library, I used the file _myenv.yml_ saved in this repository. If you decide to use _myenv.yml_, don't forget to add all your libraries in it (here we only usez azureml, scikit-learn, numpy, pandas and inference-schema).


```python
# # Create the environment file
# from azureml.core.conda_dependencies import CondaDependencies 

# myenv = CondaDependencies()
# myenv.add_conda_package("scikit-learn")

# with open("myenv.yml", "w") as f:
#     f.write(myenv.serialize_to_string())
```

### Create a configuration file

Create a deployment configuration file. Specify the number of CPUs and gigabytes of RAM needed for your Container Instances container. The cost and time of execution will then be affected by the configuration.

Here we used a 1 code and 1 Gb RAM, as for our simple use case, it is more han enough.


```python
# Create a configuration file
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "Diabetes_data",  
                                                     "method": "sklearn"},
                                               description='Predict diabetes with sklearn')
```

### Deploy as a web service

Once we have all required files, we can now deploy our model. We are now going to:
- Build an image by using these files:
   - The scoring file, _score.py_
   - The environment file, _myenv.yml_
   - The model file
- Register the image under the workspace.
- Send the image to the Container Instances container.
- Start up a container in Container Instances by using the image.
- Get the web service HTTP endpoint.


```python
%%time

# Deploy in container instances
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(runtime= "python", 
                                   entry_script="score.py",
                                   conda_file="myenv.yml")

service = Model.deploy(workspace=ws, 
                       name='diabetes-regression',
                       models=[model], 
                       inference_config=inference_config,
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)
```

    Creating service
    Running..............................................................................
    SucceededACI service creation operation finished, operation "Succeeded"
    Wall time: 6min 50s
    

We now have deployed a web service and it is ready to use in other applications. You could test whether the web service has been deployed correctly following the section __Test the deployed serice__ from this [link](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-deploy-models-with-aml#test-the-model-locally). 

We are now going to invoke this model in Power BI, in the notebook __3_Invoke_the_model_in_Power_BI__.

Remember you hve a resource group and a Container Instance services running, so that it might incur in costs in your Subscription. If you want to delete it, use the following cell to delete the service and go to the Azure portal to delete the resource group.


```python
# service.delete()
```
