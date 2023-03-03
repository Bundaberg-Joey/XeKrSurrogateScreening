# Joint AMI & surrogate libraries for screening
## Libraries Used
This repo contains code taken from two repositories under development.
The code shown is a snapshot of these libraries at the time the experiments were run.
* [ami](https://gitlab.com/AMInvestigator/ami)
* [sparse_mkl](https://github.com/Bundaberg-Joey/sparse_mkl.git)

For more information on the libraries used, contact Professor Tina Düren at the university of Bath.


## Machine Learning Tests
Tests for the machine learning code are included in `code_libs/surroagate/tests` and can be run within the docker image using the following commands:

```bash
python3 -m pytest ../code_libs/surroate/tests --runslow -s
```


## To be provided by the user
1. Directory named `cifs` containing the cif files
2. `cif_list.txt` file which lists the paths to all the cifs in the `cifs` directory. The index of the name in the file will correpsond to the indices specified in the ami output file. An example of the file contents would be:

```bash
# cif_list.txt
cifs/mof_a.cif
cifs/mof_b.cif
...
cifs/mof_zzzzz.cif
```

3. Any other code files, data inputs etc needed for the machine learning / surrogate modelling aspect of the code (i.e. feature matrix files etc)