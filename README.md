# killi-tracker


## Environment setup guide
Clone this repository or Download ZIP (green button above):


```bash
git clone https://github.com/nlammers371/killi-tracker
```


Make sure you have the following installed before starting:
- [Anaconda](https://www.anaconda.com/download)
- [Python 3.11.13](https://www.python.org/downloads/)




Navigate to the repository folder (killi-tracker) and open a Terminal (Mac/Linux) or Anaconda Prompt (Windows) and run the following commands to create a new Anaconda environment and install required packages:

Be sure to replace ```<NAME-HERE>```with a name for your environment, for example: ```killi-tracker```

```bash
conda create –name <NAME-HERE> python=3.11.13
```
```bash
conda activate <NAME-HERE>
```
```bash
conda install -c conda-forge install —file requirements.txt
```


Navigate to the ```notebooks/build_lightsheet_data.ipynb``` file and open it. Run the import cell, and then make sure you edit the following variables in the second cell in accordance with the data that you'll be working with:
- ```raw_data_root```


  - This is the path to the folder containing your raw CZI files
  - Example: (make sure it’s the entire path) _/Users/sebastian/Desktop/Research/killi_dynamics/raw_image_data/BC1/_
- ```channel_names```


   - An array of channel names used in your data
   - For example, using two channels, the variable contains: ```["BC1", "nls"]```
- ```save_root```
    - This will be the folder that all data/metadata is output from the pipeline (including zarr files)
   - Example: _/Users/sebastian/Desktop/Research/killi_dynamics/_


- ```out_name_vec```
  - Add description


After updating all variables in the first cell, you're ready to begin going through and running the following cells in the notebook. Each of these contains instructions and information about what they do.

