# Constraint-driven High-dimensional UNcompressed (Categorical) Clustering (CHUNCNet)

## Installation

### MicrOmegas
A version of the MicrOmegas software must be installed, which can be found here: [https://lapth.cnrs.fr/micromegas/](https://lapth.cnrs.fr/micromegas/).  The version used in this study was 5.2.6, which may produce different results from newer versions.

Some files in the directory must be altered, which can be done by editing the "replace_files.sh" script in the "external" directory.  The contents of the file are listed below:
```bash
#! /bin/bash
# get the directory where this script is stored
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
CHUNC_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# directory of micromegas. 
# you must change this to match.
VERSION=5.2.6
MICROMEGAS_DIR=~/physics/micromegas/micromegas_$VERSION

# copy files over
if (VERSION==5.2.6)
then
  cp $CHUNC_DIR/jake/main.c $MICROMEGAS_DIR/MSSM/
  cp $CHUNC_DIR/jake/omega.c $MICROMEGAS_DIR/sources/
  cp $CHUNC_DIR/jake/micromegas.h $MICROMEGAS_DIR/include/
else
  cp $CHUNC_DIR/main.c $MICROMEGAS_DIR/MSSM/
  cp $CHUNC_DIR/omega.c $MICROMEGAS_DIR/sources/
  cp $CHUNC_DIR/micromegas.h $MICROMEGAS_DIR/include/
fi

# run make
cd $MICROMEGAS_DIR
make clean all
make

cd MSSM
make clean all
make

cd $CHUNC_DIR
```
The micromegas version must be specified, as well as the directory of the installation.  Running this script will replace the files "main.c" in the MSSM folder, as well as "omega.c" and "micromegas.h" in the main micromegas directory.

Some common issues that can occur when compiling MicrOmegas version 5.2.6 have to due with the declaration of a variable in the file "MSSM/lib/my_complex.h".  One may get an error about multiple definitions of the variable "icomp", which can be commented out in the header file (MSSM/lib/my_complex.h):
```c++
#ifndef  __MY_COMPLEX__
#define  __MY_COMPLEX__

typedef struct{ double r;  double i; } my_complex;

//const my_complex icomp={0,1};
```
but then one will need to add its definition to the "gmuon.c" file:
```c++
/* Supersymmetric contribution to muon g-2
      
Author: A. Cottrant, june 2001             
Ref: A. Cottrant,  "Le moment anomal du muon en supersymetrie".
Rapport de DEA, Universite de Savoie.
These formulas  agree with the ones in S. Martin, J. Wells, hep-ph/0103067

Corrections:
28 march 2002: Correct factor of 1/2. in double result  line 87 
*/

#include"../../include/micromegas.h"
#include "pmodel.h"
#include "my_complex.h"
#include "matrix.h"

const my_complex icomp={0,1};
```


### SoftSUSY
A version of the SoftSUSY software must also be installed, which can be found here: [https://softsusy.hepforge.org/](https://softsusy.hepforge.org/).  The version that was used in this study is 4.1.10, which may produce different results from newer versions.  

Once both MicrOmegas and SoftSUSY are installed, the directory of both must be specified to the CHUNC generator, along with the type of parameter space one is sampling from.  An example of this is shown below:
```python
mssm = MSSMGenerator(
    microemgas_dir='~/physics/micromegas/micromegas_5.2.6/MSSM/', 
    softsusy_dir='~/physics/softsusy/softsusy-4.1.10/',
    param_space='cmssm',
)
```