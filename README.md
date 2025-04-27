### Running
Required Python version - `3.12.4`
Requirements - `pip install -r requirements.txt`

### How to run
``` Bash
$ python main.py --help
usage: main.py [-h] --input-path INPUT_PATH --output-path OUTPUT_PATH [--impute] [--select-features]

options:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH
                        Path to input csv file
  --output-path OUTPUT_PATH
                        Path to output csv file
  --impute              Perform imputation
  --select-features     Perform feature selection
```
Example
``` Bash
python main.py --input-path input.csv --output-path out.csv --impute --select-features
```