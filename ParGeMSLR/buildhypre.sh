if [ ! -d ./hypre ]
then
   git clone https://github.com/hypre-space/hypre.git;
fi
cd hypre/src;
./configure;
make -j;
