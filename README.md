# Reconstruct_inertial_current 

`python reconstruct_inertial.py`

## Regriding Croco file to rectilinear lat-lon with `regrid_croco.py`

We use surface data from the model Croco (https://www.croco-ocean.org/) on a curvilinear grid at a ~0.02° resolution.
For practical reasons and also for performance of the inversion process, we move to a rectilinear grid at a resolution of 0.1°.
To do this, I created a script that:
1) open Croco file with dask (too big for memory)
2) move side-located variables (U,V, $\tau_x$, $\tau_y$) to the mass point
3) computes geostrophic currents from SSH
4) regrid to rectilinear grid using xesmf

run 
```
python regrid_croco.py
```
 
## Building GOTM
Download files in `/home/yourname/GOTM`
```
cd ~
mkdir GOTM
cd GOTM
git clone --recursive https://github.com/gotm-model/code.git .
cd clone
git submodule update --init --recursive
```
Building into `/home/yourname/build_gotm/`
```
cd ~
mkdir build_gotm
cd build_gotm
sh /home/yourname/GOTM/code/scripts/linux/gotm_configure.sh
sh /home/yourname/GOTM/code/scripts/linux/gotm_build.sh
```

Note: you might need to change `$GOTM_BASE` in `gotm_configure.sh` to point to the right directory. 

After building, you might want to include in your `$PATH` the directory where the executable is located (indicated with `install_prefix` in `gotm_configure.sh`)

Download tests_cases (wherever you like)
```
cd ~
mkdir tests_cases_gotm
cd tests_cases_gotm
git clone https://github.com/gotm-model/cases.git .
```

running a simulation: you will need a .yml file with input. A template can be generated with `gotm --write_yaml gotm.yaml`, or you can reuse one from the test cases from above.
A case with prescribed (constant) stress is the `couette` example, a case with fluxes read from a file is the `ows_papa` case.
