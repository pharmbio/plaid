# Instructions

The .dzn layouts you want to plan should be in this directory (or the list of names should include the path to find them)

total is the number of different layouts you want to generate. For example, 4 different layouts for each file.

# Troubleshooting

If you get a permission denied error, you should execute 

```bash
$ sudo chmod 755 generate-layouts.sh
```

The script assumes that you're running it on this directory (with respect to the plate-design.mzn file). If you want to change that, change the location of that file in the script (search for ../../plate-design.mzn and give the new location).
