# NC-NET

## Config
Using template files provided and changing data to corresponding data you can use repository.

`train.py` takes one argument of name config which is yaml file which template is placed in templates directory.

## Using config file
Copy config file to main directory:
```bash
cp templates/config.yaml.template config.yaml
```

Then insert your required credentials

## Using pylint
If using linux bash use:
```bash
bash pylint.sh
```

If not install some plugin and point to pylint.rc file as configuration.

# Repository usage
A process of working with repository goes as follows:
1. Create data using prepare data script. Point to data file as input and output directory. 
At the end of process data directory should contain main_df.csv and points_df.csv files. 
2. After that adjust training process with argument and fill config file.
At the end of training the best model should be saved and tested. Results on test dataset should be printed.
3. Exporting model with export script is dependent on parameters passed in train step(points extracted).
Adjust accordingly.
