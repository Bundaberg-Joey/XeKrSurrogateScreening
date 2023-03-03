# Joint AMI & surrogate libraries for screening

## To br provided by the user
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