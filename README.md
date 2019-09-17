# Invoke a model in Power BI using Azure ML service

Power BI offers the possibility to invoke models deployed with either Azure ML service or Studio. AML services allows for more flexibility regarding model contruction regarding AML Studio. This last is however much easier to use. Follow this [link](https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-machine-learning-service/2-azure-ml-service-vs-ml-studio) to know more about AML services and Studio.

In this notebook, we are going to see how to train and deploy a model using Azure ML service and call it with Power BI. As  I particularly struggled to reach this task due to the few amount of references in the community, I decided to write this notebook to clearly explain _every_ step to call a Machine Learning model in Power BI.

To do so, we will divide this notebook in three sections:
- Model training
- Model deployment
- Invoke the deployed model in Power BI

The requirements are: 
- an Azure subscription to train & deploy your model 
- a Power BI Professional subscription to invoke your model in Power BI Dataflows
- install the azureml-sdk library. [Link](https://docs.microsoft.com/en-gb/python/api/overview/azure/ml/install?view=azure-ml-py) to the installation guide
- Python 3

Here are some resources I followed:
- https://docs.microsoft.com/en-us/power-bi/service-machine-learning-integration
- https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#example-script-with-dictionary-input-support-consumption-from-power-bi
- https://community.powerbi.com/t5/Community-Blog/Azure-Machine-Learning-in-Power-BI-Dataflows/ba-p/709744
