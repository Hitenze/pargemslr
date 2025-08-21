git clone https://github.com/Hitenze/metis_mirror
mv metis_mirror/parmetis-4.0.3.tar.gz ./
rm -rf metis_mirror/
tar -xzf parmetis-4.0.3.tar.gz
mv parmetis-4.0.3 parmetis
patch parmetis/metis/include/metis.h metis.patch
rm parmetis-4.0.3.tar.gz
cd parmetis;
make config;
make -j;
cd ..;
make -j;
