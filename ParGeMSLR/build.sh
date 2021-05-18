wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xzf parmetis-4.0.3.tar.gz
mv parmetis-4.0.3 parmetis
patch parmetis/metis/include/metis.h metis.patch
rm parmetis-4.0.3.tar.gz
cd parmetis;
make config;
make -j;
cd ..;
make -j;
