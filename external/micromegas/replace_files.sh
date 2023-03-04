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
