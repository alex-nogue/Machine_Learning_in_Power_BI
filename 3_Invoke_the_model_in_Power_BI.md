
# Invoke the deployed model in Power BI

Once we have deployed a model using Azure Machine Learning (Studio or service), one can now call this model from Power BI Dataflows.

To do so, go to [app.powerbi.com](https://app.powerbi.com), to create the Dataflow on which we are going to invoke our model. Before doing so, one must first create a Workspace to store the Dataflow.

These are the steps we are going to follow:
- __Create a Workspace__ <br>
This is where all our work is going to be packed.
- __Create a Dataflow in the Workspace__ <br>
This is where the data & model are going to stored.
- __Create a Blob Storage in Azure and upload the data__ <br>
Upload the data to the cloud to be accessible for Power BI.
- __Import the data from the Blob Storage in Dataflow__ <br>
Data on which we are going to apply the model. 
- __Invoke the model in the data just imported__ <br>

### Create a workspace

To create a workspace, simply go to app.powerbi.com and click on Workspaces as in the following image: 

<img src="pictures/workspace.png" width=200>

Then, click on "Create a new Workspace" and give a name and description to your workspace. 

### Create a Dataflow

In the new screen, click on "Get Started" under Dataflow. Should look something similat to this:

<img src="pictures/dataflow.png" width=200>

Then click on Add new entities to import your dataset. Then, select the type of your data. In our case, it is a CSV file, so that we click on:

<img src="pictures/csv.png" width=200>

In the following screen, we'll have to add the path of our data. I recommend to upload the data into a blob storage in Azure, so that it is much easier for Power BI to access to it. 

You can do it in two ways. Either from the [portal.azure.com](https://portal.azure.com) or with the AML sdk library. 

### Create a Blob Storage in Azure and upload your data

To create a Blob Storage, go to the [Azure Portal](https://portal.azure.com) and create a Storage Account. A Storage Account is where our Blob Storage is going to be held. Follow this [link](https://docs.microsoft.com/en-gb/azure/storage/common/storage-quickstart-create-account?tabs=azure-portal) to create a Storage Account.

Once you have your storage account, we can now create a Blob Storage and upload the data. To do so, run the following cell by specifying the account name and account key (_key1_) of your Storage Account. These can be found by clicking on your newly created Storage Account, as you can see in the following picture:

<img src="pictures/storage_account.png">


```python
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess

# Create a Block Blob in the Storage Account
block_blob_service = BlockBlobService(
    account_name='SET_YOURS', account_key='SET_YOURS')

# Create a Blob (container) in the Block Blob
# Here is where the data is going to be stored
container_name = 'examplename' # no special characters
block_blob_service.create_container(container_name)

# Change the access level to Blob or Container
# Otherwise, Power BI will not be able to access to the data
block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Blob)
```




    <azure.storage.blob.models.ResourceProperties at 0x1e564e2bcf8>



Now that we have our Blob, we can upload the data on it by running the following cell. The data that is going we are going to upload in our case is found in the _data_ from this repository.


```python
import os

# Change the path if your working with your own data
local_path = os.path.join(os.getcwd(),"data")
local_file_name = "diabetes.csv"
full_path_to_file = os.path.join(local_path, local_file_name)

# Upload the data in the Blob
block_blob_service.create_blob_from_path(
    container_name, local_file_name, full_path_to_file)
```




    <azure.storage.blob.models.ResourceProperties at 0x1e564e2b518>



### Import the data from the Blob Storage in Dataflow

The path of the data can be accessed via the Azure Portal. Go to Storage Accounts -> click on the Storage Account you have created -> Blobs -> click on the created container, here _examplename_ -> click on the name of your data, here _diabetes.csv_. You should have the following screen:

<img src="pictures/blob.png" width = 500>

The path of the data can be obtained from the _URL_ text box. Copy that URL and paste it on the Power BI Dataflow screen you had before:

<img src="pictures/import_dataflow.png" width=400>

Click on _Next_ and transform your table in using the Power Query editor at your convenience. Here, we just have to set the first row as header.

Then, click on _Save & Close_, set a name and refresh your Workspace. 

### Invoke the model in the data just imported

To call the model on the data, click on _Edit entities_ to access to the Power Query editor and click on _AI Insights_

<img src="pictures/AI_insights.png" width=500>

In the following screen should appear the model we have created. Here, it was called _diabetes-regression_. Verify all columns are correctly assigned and click on _Apply_. 

<img src="pictures/invoke.png" width=500>

I experienced two kind of errors here:
- The model doesn't appear: this means that either the model has not been correctly deployed, or that your Power BI subscription doesn't have access to the Azure Subscription. In the first case, review the second script __2_Model_Deplyment__. In the second case, you must grant access to your Power BI subscription as specified in the same script.
- The model appears but the variables are not displayed: the model schema has not been correctly generated. Review the second script __2_Model_Deplyment__.

Now the model has been applied, you just have to expand the columns that has been created and enjoy working with Machine Learning on Power BI!
