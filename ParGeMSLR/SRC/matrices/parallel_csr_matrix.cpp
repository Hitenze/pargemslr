
#include "../utils/memory.hpp"
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/protos.hpp"
#include "../vectors/int_vector.hpp"
#include "../vectors/sequential_vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "coo_matrix.hpp"
#include "parallel_csr_matrix.hpp"
#include "matrixops.hpp"

#include <cstring>
#ifdef PARGEMSLR_CUDA
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   CommunicationHelperClass::CommunicationHelperClass()
   {
      this->_n_in = 0;
      this->_n_out = 0;
      this->_is_ready = false;
      this->_is_waiting = false;
      this->_location_send = kMemoryHost;
      this->_location_recv = kMemoryHost;
      this->_is_nbhd_built = false;
      this->_buff_unit_size = 0;
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = 0;
#endif
      /* do nothing */
   }
   
   CommunicationHelperClass::CommunicationHelperClass(const CommunicationHelperClass &comm_helper)
   {
      if(comm_helper._is_waiting)
      {
         PARGEMSLR_ERROR("Copy a communication helper that is working. Communication helper might be changed after this job, please use move instead.");
      }
      this->_location_send = comm_helper._location_send;
      this->_location_recv = comm_helper._location_recv;
      this->_buff_unit_size = comm_helper._buff_unit_size;
      this->CreateHostBuffer(this->_buff_unit_size);
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = comm_helper._buff_d_unit_size;
      this->CreateDeviceBuffer(this->_buff_d_unit_size);
#endif
      this->_n_in = comm_helper._n_in;
      this->_n_out = comm_helper._n_out;
      this->_send_to_v = comm_helper._send_to_v;
      this->_recv_from_v = comm_helper._recv_from_v;
      this->_send_idx_v2 = comm_helper._send_idx_v2;
      this->_recv_idx_v2 = comm_helper._recv_idx_v2;
      this->_idx_helper_v2 = comm_helper._idx_helper_v2;
      /* not working, no need to copy the requrest array */
      vector<MPI_Request>().swap(this->_requests_v);
      this->_is_ready = comm_helper._is_ready;
      this->_is_waiting = comm_helper._is_waiting;
      this->_is_nbhd_built = comm_helper._is_nbhd_built;
   }
   
   CommunicationHelperClass::CommunicationHelperClass(CommunicationHelperClass &&comm_helper)
   {
      this->_location_send = comm_helper._location_send;
      comm_helper._location_send = kMemoryHost;
      this->_location_recv = comm_helper._location_recv;
      comm_helper._location_recv = kMemoryHost;
      this->_buff_unit_size = comm_helper._buff_unit_size;
      comm_helper._buff_unit_size = 0;
      this->_send_buff_v2.swap(comm_helper._send_buff_v2);
      this->_recv_buff_v2.swap(comm_helper._recv_buff_v2);
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = comm_helper._buff_d_unit_size;
      comm_helper._buff_d_unit_size = 0;
      this->_send_buff_v2_d.swap(comm_helper._send_buff_v2_d);
      this->_recv_buff_v2_d.swap(comm_helper._recv_buff_v2_d);
#endif
      this->_n_in = comm_helper._n_in;
      comm_helper._n_in = 0;
      this->_n_out = comm_helper._n_out;
      comm_helper._n_out = 0;
      this->_send_to_v = std::move(comm_helper._send_to_v);
      this->_recv_from_v = std::move(comm_helper._recv_from_v);
      this->_send_idx_v2 = std::move(comm_helper._send_idx_v2);
      this->_recv_idx_v2 = std::move(comm_helper._recv_idx_v2);
      this->_idx_helper_v2 = std::move(comm_helper._idx_helper_v2);
      /* not working, no need to copy the requrest array */
      this->_requests_v.swap(comm_helper._requests_v);
      this->_is_ready = comm_helper._is_ready;
      comm_helper._is_ready = false;
      this->_is_waiting = comm_helper._is_waiting;
      comm_helper._is_waiting = false;
      this->_is_nbhd_built = comm_helper._is_nbhd_built;
      comm_helper._is_nbhd_built = false;
   }
   
   CommunicationHelperClass& CommunicationHelperClass::operator= (const CommunicationHelperClass &comm_helper)
   {
      this->Clear();
      if(comm_helper._is_waiting)
      {
         PARGEMSLR_ERROR("Using operator = on a communication helper that is working. Communication helper might be changed after this job, please use move instead.");
      }
      this->_location_send = comm_helper._location_send;
      this->_location_recv = comm_helper._location_recv;
      this->_buff_unit_size = comm_helper._buff_unit_size;
      this->CreateHostBuffer(this->_buff_unit_size);
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = comm_helper._buff_d_unit_size;
      this->CreateDeviceBuffer(this->_buff_d_unit_size);
#endif
      this->_n_in = comm_helper._n_in;
      this->_n_out = comm_helper._n_out;
      this->_send_to_v = comm_helper._send_to_v;
      this->_recv_from_v = comm_helper._recv_from_v;
      this->_send_idx_v2 = comm_helper._send_idx_v2;
      this->_recv_idx_v2 = comm_helper._recv_idx_v2;
      this->_idx_helper_v2 = comm_helper._idx_helper_v2;
      /* not working, no need to copy the requrest array */
      vector<MPI_Request>().swap(this->_requests_v);
      this->_is_ready = comm_helper._is_ready;
      this->_is_waiting = comm_helper._is_waiting;
      this->_is_nbhd_built = comm_helper._is_nbhd_built;
      return *this;
   }
   
   CommunicationHelperClass& CommunicationHelperClass::operator= (CommunicationHelperClass &&comm_helper)
   {
      this->Clear();
      this->_location_send = comm_helper._location_send;
      comm_helper._location_send = kMemoryHost;
      this->_location_recv = comm_helper._location_recv;
      comm_helper._location_recv = kMemoryHost;
      this->_buff_unit_size = comm_helper._buff_unit_size;
      comm_helper._buff_unit_size = 0;
      this->_send_buff_v2.swap(comm_helper._send_buff_v2);
      this->_recv_buff_v2.swap(comm_helper._recv_buff_v2);
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = comm_helper._buff_d_unit_size;
      comm_helper._buff_d_unit_size = 0;
      this->_send_buff_v2_d.swap(comm_helper._send_buff_v2_d);
      this->_recv_buff_v2_d.swap(comm_helper._recv_buff_v2_d);
#endif
      this->_n_in = comm_helper._n_in;
      comm_helper._n_in = 0;
      this->_n_out = comm_helper._n_out;
      comm_helper._n_out = 0;
      this->_send_to_v = std::move(comm_helper._send_to_v);
      this->_recv_from_v = std::move(comm_helper._recv_from_v);
      this->_send_idx_v2.swap(comm_helper._send_idx_v2);
      this->_recv_idx_v2.swap(comm_helper._recv_idx_v2);
      this->_idx_helper_v2.swap(comm_helper._idx_helper_v2);
      /* not working, no need to copy the requrest array */
      this->_requests_v.swap(comm_helper._requests_v);
      this->_is_ready = comm_helper._is_ready;
      comm_helper._is_ready = false;
      this->_is_waiting = comm_helper._is_waiting;
      comm_helper._is_waiting = false;
      this->_is_nbhd_built = comm_helper._is_nbhd_built;
      comm_helper._is_nbhd_built = false;
      return *this;
   }
   
   CommunicationHelperClass::~CommunicationHelperClass()
   {
      this->Clear();
   }
   
   int CommunicationHelperClass::Clear()
   {
      if(this->_is_waiting)
      {
         PARGEMSLR_ERROR("Tring to destroy a working communication helper.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      /* call base clear function */
      parallel_log::Clear();
      
      this->_n_in = 0;
      this->_n_out = 0;
      this->_is_ready = false;
      this->_is_waiting = false;
      this->_is_nbhd_built = false;
      this->_location_send = kMemoryHost;
      this->_location_recv = kMemoryHost;
      this->_buff_unit_size = 0;
#ifdef PARGEMSLR_CUDA
      this->_buff_d_unit_size = 0;
#endif
      
      size_t i;
      
      this->_send_to_v.Clear();
      this->_recv_from_v.Clear();
      
      vector<MPI_Request>().swap(this->_requests_v);
      
      for(i = 0 ; i < this->_send_idx_v2.size() ; i ++)
      {
         this->_send_idx_v2[i].Clear();
      }
      std::vector<vector_int >().swap(this->_send_idx_v2);
      
      for(i = 0 ; i < this->_recv_idx_v2.size() ; i ++)
      {
         this->_recv_idx_v2[i].Clear();
      }
      std::vector<vector_int >().swap(this->_recv_idx_v2);
      
      for(i = 0 ; i < this->_idx_helper_v2.size() ; i ++)
      {
         this->_idx_helper_v2[i].Clear();
      }
      std::vector<vector_int >().swap(this->_idx_helper_v2);
      
      for(i = 0 ; i < this->_send_buff_v2.size() ; i ++)
      {
         PARGEMSLR_FREE( this->_send_buff_v2[i], kMemoryHost );
      }
      std::vector<void*>().swap(this->_send_buff_v2);
      
      for(i = 0 ; i < this->_recv_buff_v2.size() ; i ++)
      {
         PARGEMSLR_FREE( this->_recv_buff_v2[i], kMemoryHost );
      }
      std::vector<void*>().swap(this->_recv_buff_v2);
      
#ifdef PARGEMSLR_CUDA
      
      for(i = 0 ; i < this->_send_buff_v2_d.size() ; i ++)
      {
         PARGEMSLR_FREE( this->_send_buff_v2_d[i], kMemoryDevice );
      }
      std::vector<void*>().swap(this->_send_buff_v2_d);
      
      for(i = 0 ; i < this->_recv_buff_v2_d.size() ; i ++)
      {
         PARGEMSLR_FREE( this->_recv_buff_v2_d[i], kMemoryDevice );
      }
      std::vector<void*>().swap(this->_recv_buff_v2_d);
      
#endif
      
      return PARGEMSLR_SUCCESS;
   }
   
   int CommunicationHelperClass::MoveData(int location)
   {
      /* note that the buffer won't be touched here */
      this->MoveSendData(location);
      this->MoveRecvData(location);
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CommunicationHelperClass::MoveSendData(int location)
   {
      /* note that the buffer won't be touched here */
      if(location == this->_location_send)
      {
         return PARGEMSLR_SUCCESS;
      }
      this->_location_send = location;
      
      size_t i;
      
      for(i = 0 ; i < _send_idx_v2.size() ; i ++)
      {
         _send_idx_v2[i].MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CommunicationHelperClass::MoveRecvData(int location)
   {
      /* note that the buffer won't be touched here */
      if(location == this->_location_recv)
      {
         return PARGEMSLR_SUCCESS;
      }
      this->_location_recv = location;
      
      size_t i;
      
      for(i = 0 ; i < _recv_idx_v2.size() ; i ++)
      {
         _recv_idx_v2[i].MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
      
   }
   
   int CommunicationHelperClass::CreateHostBuffer(int unitsize)
   {
      PARGEMSLR_CHKERR(unitsize < 0);
      if(unitsize == 0 || unitsize <= this->_buff_unit_size)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int      i, nsends, nrecvs, length;
      size_t   old_size, new_size;
      
      /* start creating buffer */
      if( this->_buff_unit_size == 0 )
      {
         /* in this case, we haven't created the buffer yet */
         nsends = (int) this->_send_idx_v2.size();
         nrecvs = (int) this->_recv_idx_v2.size();
         
         this->_send_buff_v2.resize(nsends);
         this->_recv_buff_v2.resize(nrecvs);
         for(i = 0 ; i < nsends ; i ++)
         {
            length = this->_send_idx_v2[i].GetLengthLocal();
            new_size = unitsize * length;
            PARGEMSLR_MALLOC_VOID( this->_send_buff_v2[i], new_size, kMemoryHost);
         }
         
         for(i = 0 ; i < nrecvs ; i ++)
         {
            length = this->_recv_idx_v2[i].GetLengthLocal();
            new_size = unitsize * length;
            PARGEMSLR_MALLOC_VOID( this->_recv_buff_v2[i], new_size, kMemoryHost);
         }
      }
      else
      {
         /* in this case, we already have the buffer, check if the memory is enough */
         nsends = (int) this->_send_idx_v2.size();
         nrecvs = (int) this->_recv_idx_v2.size();
         
         for(i = 0 ; i < nsends ; i ++)
         {
            length = this->_send_idx_v2[i].GetLengthLocal();
            old_size = this->_buff_unit_size * length;
            new_size = unitsize * length;
            PARGEMSLR_REALLOC_VOID( this->_send_buff_v2[i], old_size, new_size, kMemoryHost);
         }
         
         for(i = 0 ; i < nrecvs ; i ++)
         {
            length = this->_recv_idx_v2[i].GetLengthLocal();
            old_size = this->_buff_unit_size * length;
            new_size = unitsize * length;
            PARGEMSLR_REALLOC_VOID( this->_recv_buff_v2[i], old_size, new_size, kMemoryHost);
         }
      }
      
      /* update buffer size */
      this->_buff_unit_size = unitsize;
      
      return PARGEMSLR_SUCCESS;
   }

#ifdef PARGEMSLR_CUDA
   int CommunicationHelperClass::CreateDeviceBuffer(int unitsize)
   {
      PARGEMSLR_CHKERR(unitsize < 0);
      if(unitsize == 0 || unitsize <= this->_buff_d_unit_size)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int      i, nsends, nrecvs, length;
      size_t   old_size, new_size;
      
      /* start creating buffer */
      if( this->_buff_d_unit_size == 0 )
      {
         /* in this case, we haven't created the buffer yet */
         nsends = (int) this->_send_idx_v2.size();
         nrecvs = (int) this->_recv_idx_v2.size();
         
         this->_send_buff_v2_d.resize(nsends);
         this->_recv_buff_v2_d.resize(nrecvs);
         for(i = 0 ; i < nsends ; i ++)
         {
            length = this->_send_idx_v2[i].GetLengthLocal();
            new_size = unitsize * length;
            PARGEMSLR_MALLOC_VOID( this->_send_buff_v2_d[i], new_size, kMemoryDevice);
         }
         
         for(i = 0 ; i < nrecvs ; i ++)
         {
            length = this->_recv_idx_v2[i].GetLengthLocal();
            new_size = unitsize * length;
            PARGEMSLR_MALLOC_VOID( this->_recv_buff_v2_d[i], new_size, kMemoryDevice);
         }
      }
      else
      {
         /* in this case, we already have the buffer, check if the memory is enough */
         nsends = (int) this->_send_idx_v2.size();
         nrecvs = (int) this->_recv_idx_v2.size();
         
         for(i = 0 ; i < nsends ; i ++)
         {
            length = this->_send_idx_v2[i].GetLengthLocal();
            old_size = this->_buff_d_unit_size * length;
            new_size = unitsize * length;
            PARGEMSLR_REALLOC_VOID( this->_send_buff_v2_d[i], old_size, new_size, kMemoryDevice);
         }
         
         for(i = 0 ; i < nrecvs ; i ++)
         {
            length = this->_recv_idx_v2[i].GetLengthLocal();
            old_size = this->_buff_d_unit_size * length;
            new_size = unitsize * length;
            PARGEMSLR_REALLOC_VOID( this->_recv_buff_v2_d[i], old_size, new_size, kMemoryDevice);
         }
      }
      
      /* update buffer size */
      this->_buff_d_unit_size = unitsize;
      
      return PARGEMSLR_SUCCESS;
   }
#endif
   
   template <typename T>
   int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<T> &vec_in, VectorVirtualClass<T> &vec_out, int loc_in, int loc_out)
   {
      this->DataTransferStart(vec_in, loc_in);
      this->DataTransferOver(vec_out, loc_out);
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<int> &vec_in, VectorVirtualClass<int> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<long int> &vec_in, VectorVirtualClass<long int> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<float> &vec_in, VectorVirtualClass<float> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<double> &vec_in, VectorVirtualClass<double> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<complexs> &vec_in, VectorVirtualClass<complexs> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer(const VectorVirtualClass<complexd> &vec_in, VectorVirtualClass<complexd> &vec_out, int loc_in, int loc_out);
   
   template <typename T>
   int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<T> &vec_in, int loc_in)
   {
      if(!(this->_is_ready))
      {
         PARGEMSLR_ERROR("Communication helper not ready, please call SetupMatvec() function first.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /* stop if some other communication is ongoing */
      PARGEMSLR_CHKERR(this->_is_waiting);
      this->_is_waiting = true;
      
      PARGEMSLR_CHKERR(vec_in.GetLengthLocal() != this->_n_in);
      
      int   i, j, nsends, nrecvs, unitsize, length;
      
      /* for integer, create integer vector, otherwise real vector */
      typename std::conditional<PargemslrIsInteger<T>::value, IntVectorClass<T>, SequentialVectorClass<T>>::type send_ptr;
      
      /* get MPI info */
      MPI_Comm comm;
      int      np, myid;
      this->GetMpiInfo(np, myid, comm);
      
      /* start copying data */
      nsends = (int) this->_send_idx_v2.size();
      nrecvs = (int) this->_recv_idx_v2.size();
      
      this->_requests_v.resize(nsends+nrecvs);
      
      /* This step is to check the loaction of the input data
       * We might need to handle same input data for multiple times, thus,
       * we move the indices vectors.
       * Check location only when cuda enabled.
       */
#ifdef PARGEMSLR_CUDA
      switch(loc_in)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the input data is on the host */
            if(this->_location_send == kMemoryDevice)
            {
               /* moving buffer to the host */
               this->MoveSendData(kMemoryHost);
            }
            break;
         }
         case kMemoryDevice:
         {
            /* the input data is on the device memory */
            if(this->_location_send == kMemoryHost || this->_location_send == kMemoryPinned)
            {
               /* moving buffer to the device */
               this->MoveSendData(kMemoryDevice);
            }
            break;
         }
         default:
         {
            /* unified memory, do nothing */
            break;
         }
      }
#endif

      /* Now start the communication
       * v_in on host: host perm
       * v_in on device: device perm
       */
      switch(loc_in)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the input data is on the host */
            
            /* update host buffer */
            unitsize = sizeof(T);
            this->CreateHostBuffer(unitsize);
            
            /* apply permutation on the host */
            for(i = 0 ; i < nsends ; i ++)
            {
               length = this->_send_idx_v2[i].GetLengthLocal();
               send_ptr.SetupPtr( (T*) (this->_send_buff_v2[i]), length, kMemoryHost);
               this->_send_idx_v2[i].GatherPerm(vec_in, send_ptr);
            }
            
            break;
         }
#ifdef PARGEMSLR_CUDA
         case kMemoryDevice: case kMemoryUnified:
         {
            /* the input data is on the device memory or unified memory, apply perm on the device */
            
            /* update all buffers */
            unitsize = sizeof(T);
            this->CreateHostBuffer(unitsize);
            this->CreateDeviceBuffer(unitsize);
            /* apply permutation on the device, and copy result from device to host */
            for(i = 0 ; i < nsends ; i ++)
            {
               length = this->_send_idx_v2[i].GetLengthLocal();
               send_ptr.SetupPtr( (T*) (this->_send_buff_v2_d[i]), length, kMemoryDevice);
               this->_send_idx_v2[i].GatherPerm(vec_in, send_ptr);
               PARGEMSLR_MEMCPY(this->_send_buff_v2[i], this->_send_buff_v2_d[i], length, kMemoryHost, kMemoryDevice, T);
            }
            
            break;
         }
#endif
         default:
         {
            /* should not reach here */
            PARGEMSLR_ERROR("Unknown memory location.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }

      
      /* mpi send and recv on host buffer */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( (T*) (this->_send_buff_v2[i]), this->_send_idx_v2[i].GetLengthLocal(), 
                           this->_send_to_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( (T*) (this->_recv_buff_v2[i]), this->_recv_idx_v2[i].GetLengthLocal(),
                           this->_recv_from_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<int> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<long int> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<float> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<double> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<complexs> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart(const VectorVirtualClass<complexd> &vec_in, int loc_in);
   
   template <typename T>
   int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<T> &vec_out, int loc_out)
   {
      if(!(this->_is_waiting))
      {
         return PARGEMSLR_SUCCESS;
      }
      
      PARGEMSLR_CHKERR(vec_out.GetLengthLocal() != this->_n_out);
      
      int   i, sendrecvs, nsends, nrecvs, idx, length;
      
      /* for integer, create integer vector, otherwise real vector */
      typename std::conditional<PargemslrIsInteger<T>::value, IntVectorClass<T>, SequentialVectorClass<T>>::type recv_ptr;
      
      /* start recing data 
       * note that vec_out typicaly should have same location as 
       * vec_in, however, we need to support different location.
       * If vec_in and vec_out has different location, the index 
       * vector will follow the location of vec_in.
       * We definitly can have those indices on different location,
       * however, since this is not common, we don't do it.
       */
      
      nsends = (int) this->_send_idx_v2.size();
      nrecvs = (int) this->_recv_idx_v2.size();
      sendrecvs = nsends + nrecvs;
      
      /* This step is to check the loaction of the output data
       * We might need to handle same output data for multiple times, thus,
       * we move the indices vectors.
       */
#ifdef PARGEMSLR_CUDA
      switch(loc_out)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the output data is on the host */
            if(this->_location_recv == kMemoryDevice)
            {
               /* moving buffer to the host */
               this->MoveRecvData(kMemoryHost);
            }
            break;
         }
         case kMemoryDevice:
         {
            /* the output data is on the device memory */
            if(this->_location_recv == kMemoryHost || this->_location_recv == kMemoryPinned)
            {
               /* moving buffer to the device */
               this->MoveRecvData(kMemoryDevice);
            }
            break;
         }
         default:
         {
            /* unified memory, do nothing */
            break;
         }
      }
#endif

      /* now apply permutation and copy data */
      switch(loc_out)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* on the host */
            for(i = 0 ; i < sendrecvs ; i ++)
            {
               /* wait for any communication to be finished */
               PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecvs, this->_requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
               if(idx >= nsends)
               {
                  /* this is a recv that is finished */
                  idx -= nsends;
                  length = this->_recv_idx_v2[idx].GetLengthLocal();
                  recv_ptr.SetupPtr( (T*) this->_recv_buff_v2[idx], length, kMemoryHost);
                  this->_recv_idx_v2[idx].ScatterRperm(recv_ptr, vec_out);
               } 
            }
            break;
         }
#ifdef PARGEMSLR_CUDA
         case kMemoryDevice: case kMemoryUnified:
         {
            /* on the device */
            for(i = 0 ; i < sendrecvs ; i ++)
            {
               /* wait for any communication to be finished */
               PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecvs, this->_requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
               
               if(idx >= nsends)
               {
                  /* this is a recv that is finished */
                  idx -= nsends;
                  length = this->_recv_idx_v2[idx].GetLengthLocal();
                  /* copy to device */
                  PARGEMSLR_MEMCPY(this->_recv_buff_v2_d[idx], this->_recv_buff_v2[idx], length, kMemoryDevice, kMemoryHost, T);
                  /* apply device permutation */
                  recv_ptr.SetupPtr( (T*) this->_recv_buff_v2_d[idx], length, kMemoryDevice);
                  this->_recv_idx_v2[idx].ScatterRperm(recv_ptr, vec_out);
               } 
            }
            break;
         }
#endif
         default:
         {
            /* should not reach here */
            PARGEMSLR_ERROR("Unknown memory location.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }
      
      this->_is_waiting = false;
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<int> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<long int> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<float> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<double> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<complexs> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver(VectorVirtualClass<complexd> &vec_out, int loc_out);
   
   template <typename T, class VectorType>
   int CommunicationHelperClass::DataTransfer(const std::vector<VectorType> &vec_in, std::vector<VectorType> &vec_out, int loc_in, int loc_out)
   {
      this->DataTransferStart<T, VectorType>(vec_in, loc_in);
      this->DataTransferOver<T, VectorType>(vec_out, loc_out);
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransfer<int, vector_int>(const std::vector<vector_int> &vec_in, std::vector<vector_int> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer<long int, vector_long>(const std::vector<vector_long> &vec_in, std::vector<vector_long> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer<float, vector_seq_float>(const std::vector<vector_seq_float> &vec_in, std::vector<vector_seq_float> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer<double, vector_seq_double>(const std::vector<vector_seq_double> &vec_in, std::vector<vector_seq_double> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer<complexs, vector_seq_complexs>(const std::vector<vector_seq_complexs> &vec_in, std::vector<vector_seq_complexs> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransfer<complexd, vector_seq_complexd>(const std::vector<vector_seq_complexd> &vec_in, std::vector<vector_seq_complexd> &vec_out, int loc_in, int loc_out);
   
   template <typename T, class VectorType>
   int CommunicationHelperClass::DataTransferStart(const std::vector<VectorType> &vec_in, int loc_in)
   {
      if(!(this->_is_ready))
      {
         PARGEMSLR_ERROR("Communication helper not ready, please call SetupMatvec() function first.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /* stop if some other communication is ongoing */
      PARGEMSLR_CHKERR(this->_is_waiting);
      this->_is_waiting = true;
      
      int   i, j, nsends, nrecvs, unitsize;
      
      /* get MPI info */
      MPI_Comm comm;
      int      np, myid;
      this->GetMpiInfo(np, myid, comm);
      
      /* start copying data */
      nsends = (int) this->_send_idx_v2.size();
      nrecvs = (int) this->_recv_idx_v2.size();
      
      this->_requests_v.resize(nsends+nrecvs);
      
      /* now check the size, location of the input data */
      PARGEMSLR_CHKERR((int)(vec_in.size()) != nsends);
      
      /* Now start the communication */
      
      /* update host buffer */
      unitsize = sizeof(T);
      this->CreateHostBuffer(unitsize);
      
      /* mpi send and recv on host buffer */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         /* copy data into send buffer */
         PARGEMSLR_MEMCPY( this->_send_buff_v2[i], vec_in[i].GetData(), this->_send_idx_v2[i].GetLengthLocal(), kMemoryHost, loc_in, T);
         
         /* apply the send */
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( (T*) (this->_send_buff_v2[i]), this->_send_idx_v2[i].GetLengthLocal(), 
                           this->_send_to_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( (T*) (this->_recv_buff_v2[i]), this->_recv_idx_v2[i].GetLengthLocal(),
                           this->_recv_from_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferStart<int, vector_int>(const std::vector<vector_int> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart<long int, vector_long>(const std::vector<vector_long> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart<float, vector_seq_float>(const std::vector<vector_seq_float> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart<double, vector_seq_double>(const std::vector<vector_seq_double> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart<complexs, vector_seq_complexs>(const std::vector<vector_seq_complexs> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStart<complexd, vector_seq_complexd>(const std::vector<vector_seq_complexd> &vec_in, int loc_in);
   
   template <typename T, class VectorType>
   int CommunicationHelperClass::DataTransferOver(std::vector<VectorType> &vec_out, int loc_out)
   {
      if(!(this->_is_waiting))
      {
         return PARGEMSLR_SUCCESS;
      }
      
      int   i, sendrecvs, nsends, nrecvs, idx;
      
      /* start recing data */
      
      nsends = (int) this->_send_idx_v2.size();
      nrecvs = (int) this->_recv_idx_v2.size();
      sendrecvs = nsends + nrecvs;
      
      /* now check the size, location of the input data */
      PARGEMSLR_CHKERR((int)(vec_out.size()) != nrecvs);
      
      /* now recv and copy data */
      for(i = 0 ; i < sendrecvs ; i ++)
      {
         /* wait for any communication to be finished */
         PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecvs, this->_requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
         if(idx >= nsends)
         {
            /* this is a recv that is finished
             * copy data from recv buffer 
             */
            idx -= nsends;
            PARGEMSLR_MEMCPY( vec_out[idx].GetData(), this->_recv_buff_v2[idx], this->_recv_idx_v2[idx].GetLengthLocal(), loc_out, kMemoryHost, T);
         } 
      }
      
      this->_is_waiting = false;
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferOver<int, vector_int>(std::vector<vector_int> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver<long int, vector_long>(std::vector<vector_long> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver<float, vector_seq_float>(std::vector<vector_seq_float> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver<double, vector_seq_double>(std::vector<vector_seq_double> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver<complexs, vector_seq_complexs>(std::vector<vector_seq_complexs> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOver<complexd, vector_seq_complexd>(std::vector<vector_seq_complexd> &vec_out, int loc_out);
   
   template <typename T>
   int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<T> &vec_in, VectorVirtualClass<T> &vec_out, int loc_in, int loc_out)
   {
      this->DataTransferStartReverse(vec_in, loc_in);
      this->DataTransferOverReverse(vec_out, loc_out);
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<int> &vec_in, VectorVirtualClass<int> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<long int> &vec_in, VectorVirtualClass<long int> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<float> &vec_in, VectorVirtualClass<float> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<double> &vec_in, VectorVirtualClass<double> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<complexs> &vec_in, VectorVirtualClass<complexs> &vec_out, int loc_in, int loc_out);
   template int CommunicationHelperClass::DataTransferReverse(const VectorVirtualClass<complexd> &vec_in, VectorVirtualClass<complexd> &vec_out, int loc_in, int loc_out);
   
   template <typename T>
   int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<T> &vec_in, int loc_in)
   {
      if(!(this->_is_ready))
      {
         PARGEMSLR_ERROR("Communication helper not ready, please call SetupMatvec() function first.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      /* stop if some other communication is ongoing */
      PARGEMSLR_CHKERR(this->_is_waiting);
      this->_is_waiting = true;
      
      PARGEMSLR_CHKERR(vec_in.GetLengthLocal() != this->_n_out);
      
      int   i, j, nsends, nrecvs, unitsize, length;
      
      /* for integer, create integer vector, otherwise real vector */
      typename std::conditional<PargemslrIsInteger<T>::value, IntVectorClass<T>, SequentialVectorClass<T>>::type send_ptr;
      
      /* get MPI info */
      MPI_Comm comm;
      int      np, myid;
      this->GetMpiInfo(np, myid, comm);
      
      /* start copying data */
      nsends = (int) this->_recv_idx_v2.size();
      nrecvs = (int) this->_send_idx_v2.size();
      
      this->_requests_v.resize(nsends+nrecvs);
      
      /* This step is to check the loaction of the input data
       * We might need to handle same input data for multiple times, thus,
       * we move the indices vectors.
       * Check location only when cuda enabled.
       */
#ifdef PARGEMSLR_CUDA
      switch(loc_in)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the input data is on the host */
            if(this->_location_recv == kMemoryDevice)
            {
               /* moving buffer to the host */
               this->MoveRecvData(kMemoryHost);
            }
            break;
         }
         case kMemoryDevice:
         {
            /* the input data is on the device memory */
            if(this->_location_recv == kMemoryHost || this->_location_recv == kMemoryPinned)
            {
               /* moving buffer to the device */
               this->MoveRecvData(kMemoryDevice);
            }
            break;
         }
         default:
         {
            /* unified memory, do nothing */
            break;
         }
      }
#endif

      /* Now start the communication
       * v_in on host: host perm
       * v_in on device: device perm
       */
      switch(loc_in)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the input data is on the host */
            
            /* update host buffer */
            unitsize = sizeof(T);
            this->CreateHostBuffer(unitsize);
            
            /* apply permutation on the host */
            for(i = 0 ; i < nsends ; i ++)
            {
               length = this->_recv_idx_v2[i].GetLengthLocal();
               send_ptr.SetupPtr( (T*) (this->_recv_buff_v2[i]), length, kMemoryHost);
               this->_recv_idx_v2[i].GatherPerm(vec_in, send_ptr);
            }
            
            break;
         }
#ifdef PARGEMSLR_CUDA
         case kMemoryDevice: case kMemoryUnified:
         {
            /* the input data is on the device memory or unified memory, apply perm on the device */
            
            /* update all buffers */
            unitsize = sizeof(T);
            this->CreateHostBuffer(unitsize);
            this->CreateDeviceBuffer(unitsize);
            /* apply permutation on the device, and copy result from device to host */
            for(i = 0 ; i < nsends ; i ++)
            {
               length = this->_recv_idx_v2[i].GetLengthLocal();
               send_ptr.SetupPtr( (T*) (this->_recv_buff_v2_d[i]), length, kMemoryDevice);
               this->_recv_idx_v2[i].GatherPerm(vec_in, send_ptr);
               PARGEMSLR_MEMCPY(this->_recv_buff_v2[i], this->_recv_buff_v2_d[i], length, kMemoryHost, kMemoryDevice, T);
            }
            
            break;
         }
#endif
         default:
         {
            /* should not reach here */
            PARGEMSLR_ERROR("Unknown memory location.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }

      
      /* mpi send and recv on host buffer */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIsend( (T*) (this->_recv_buff_v2[i]), this->_recv_idx_v2[i].GetLengthLocal(), 
                           this->_recv_from_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( (T*) (this->_send_buff_v2[i]), this->_send_idx_v2[i].GetLengthLocal(),
                           this->_send_to_v[i], 0, comm, &(this->_requests_v[j++])) );
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<int> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<long int> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<float> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<double> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<complexs> &vec_in, int loc_in);
   template int CommunicationHelperClass::DataTransferStartReverse(const VectorVirtualClass<complexd> &vec_in, int loc_in);
   
   template <typename T>
   int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<T> &vec_out, int loc_out)
   {
      if(!(this->_is_waiting))
      {
         return PARGEMSLR_SUCCESS;
      }
      
      PARGEMSLR_CHKERR(vec_out.GetLengthLocal() != this->_n_in);
      
      int   i, sendrecvs, nsends, nrecvs, idx, length;
      
      /* for integer, create integer vector, otherwise real vector */
      typename std::conditional<PargemslrIsInteger<T>::value, IntVectorClass<T>, SequentialVectorClass<T>>::type recv_ptr;
      
      nsends = (int) this->_recv_idx_v2.size();
      nrecvs = (int) this->_send_idx_v2.size();
      sendrecvs = nsends + nrecvs;
      
      /* This step is to check the loaction of the output data
       * We might need to handle same output data for multiple times, thus,
       * we move the indices vectors.
       */
#ifdef PARGEMSLR_CUDA
      switch(loc_out)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* the output data is on the host */
            if(this->_location_send == kMemoryDevice)
            {
               /* moving buffer to the host */
               this->MoveSendData(kMemoryHost);
            }
            break;
         }
         case kMemoryDevice:
         {
            /* the output data is on the device memory */
            if(this->_location_send == kMemoryHost || this->_location_send == kMemoryPinned)
            {
               /* moving buffer to the device */
               this->MoveSendData(kMemoryDevice);
            }
            break;
         }
         default:
         {
            /* unified memory, do nothing */
            break;
         }
      }
#endif

      /* now apply permutation and copy data */
      switch(loc_out)
      {
         case kMemoryHost: case kMemoryPinned:
         {
            /* on the host */
            for(i = 0 ; i < sendrecvs ; i ++)
            {
               /* wait for any communication to be finished */
               PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecvs, this->_requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
               if(idx >= nsends)
               {
                  /* this is a recv that is finished */
                  idx -= nsends;
                  length = this->_send_idx_v2[idx].GetLengthLocal();
                  recv_ptr.SetupPtr( (T*) this->_send_buff_v2[idx], length, kMemoryHost);
                  this->_send_idx_v2[idx].ScatterRperm(recv_ptr, vec_out);
               } 
            }
            break;
         }
#ifdef PARGEMSLR_CUDA
         case kMemoryDevice: case kMemoryUnified:
         {
            /* on the device */
            for(i = 0 ; i < sendrecvs ; i ++)
            {
               /* wait for any communication to be finished */
               PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecvs, this->_requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
               
               if(idx >= nsends)
               {
                  /* this is a recv that is finished */
                  idx -= nsends;
                  length = this->_send_idx_v2[idx].GetLengthLocal();
                  /* copy to device */
                  PARGEMSLR_MEMCPY(this->_send_buff_v2_d[idx], this->_send_buff_v2[idx], length, kMemoryDevice, kMemoryHost, T);
                  /* apply device permutation */
                  recv_ptr.SetupPtr( (T*) this->_send_buff_v2_d[idx], length, kMemoryDevice);
                  this->_send_idx_v2[idx].ScatterRperm(recv_ptr, vec_out);
               } 
            }
            break;
         }
#endif
         default:
         {
            /* should not reach here */
            PARGEMSLR_ERROR("Unknown memory location.");
            return PARGEMSLR_ERROR_INVALED_PARAM;
         }
      }
      
      this->_is_waiting = false;
      
      return PARGEMSLR_SUCCESS;
   }
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<int> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<long int> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<float> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<double> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<complexs> &vec_out, int loc_out);
   template int CommunicationHelperClass::DataTransferOverReverse(VectorVirtualClass<complexd> &vec_out, int loc_out);
   
   template <typename T>
   ParallelCsrMatrixClass<T>::ParallelCsrMatrixClass()
   {
      _nrow_global = 0;
      _ncol_global = 0;
      _nrow_start = 0;
      _ncol_start = 0;
      _nrow_local = 0;
      _ncol_local = 0;
      _is_offd_map_sorted = false;
      _separator_ndom = 0;
   }
   template ParallelCsrMatrixClass<float>::ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<double>::ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<complexs>::ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<complexd>::ParallelCsrMatrixClass();
   
   template <typename T>
   ParallelCsrMatrixClass<T>::ParallelCsrMatrixClass(const ParallelCsrMatrixClass<T> &mat) : MatrixClass<T>(mat)
   {
      this->_nrow_global = mat._nrow_global;
      this->_ncol_global = mat._ncol_global;
      this->_nrow_start = mat._nrow_start;
      this->_ncol_start = mat._ncol_start;
      this->_nrow_local = mat._nrow_local;
      this->_ncol_local = mat._ncol_local;
      this->_is_offd_map_sorted = mat._is_offd_map_sorted;
      this->_diag_mat = mat._diag_mat;
      this->_offd_mat = mat._offd_mat;
      this->_offd_map_v = mat._offd_map_v;
      this->_matvec_working_vec = mat._matvec_working_vec;
      this->_trans_matvec_working_vec = mat._trans_matvec_working_vec;
      this->_comm_helper = mat._comm_helper;
      this->_trans_comm_helper = mat._trans_comm_helper;
      this->_separator_ndom = mat._separator_ndom;
      this->_separator_domi = mat._separator_domi;
   }
   template ParallelCsrMatrixClass<float>::ParallelCsrMatrixClass(const ParallelCsrMatrixClass<float> &mat);
   template ParallelCsrMatrixClass<double>::ParallelCsrMatrixClass(const ParallelCsrMatrixClass<double> &mat);
   template ParallelCsrMatrixClass<complexs>::ParallelCsrMatrixClass(const ParallelCsrMatrixClass<complexs> &mat);
   template ParallelCsrMatrixClass<complexd>::ParallelCsrMatrixClass(const ParallelCsrMatrixClass<complexd> &mat);
   
   template <typename T>
   ParallelCsrMatrixClass<T>::ParallelCsrMatrixClass( ParallelCsrMatrixClass<T> &&mat) : MatrixClass<T>(std::move(mat))
   {
      this->_nrow_global = mat._nrow_global;
      mat._nrow_global = 0;
      this->_ncol_global = mat._ncol_global;
      mat._ncol_global = 0;
      this->_nrow_start = mat._nrow_start;
      mat._nrow_start = 0;
      this->_ncol_start = mat._ncol_start;
      mat._ncol_start = 0;
      this->_nrow_local = mat._nrow_local;
      mat._nrow_local = 0;
      this->_ncol_local = mat._ncol_local;
      mat._ncol_local = 0;
      this->_is_offd_map_sorted = mat._is_offd_map_sorted;
      mat._is_offd_map_sorted = false;
      this->_diag_mat = std::move(mat._diag_mat);
      this->_offd_mat = std::move(mat._offd_mat);
      this->_offd_map_v = std::move(mat._offd_map_v);
      this->_matvec_working_vec = std::move(mat._matvec_working_vec);
      this->_trans_matvec_working_vec = std::move(mat._trans_matvec_working_vec);
      this->_comm_helper = std::move(mat._comm_helper);
      this->_trans_comm_helper = std::move(mat._trans_comm_helper);
      this->_separator_ndom = mat._separator_ndom;
      mat._separator_ndom = 0;
      this->_separator_domi = std::move(mat._separator_domi);
   }
   template ParallelCsrMatrixClass<float>::ParallelCsrMatrixClass( ParallelCsrMatrixClass<float> &&mat);
   template ParallelCsrMatrixClass<double>::ParallelCsrMatrixClass( ParallelCsrMatrixClass<double> &&mat);
   template ParallelCsrMatrixClass<complexs>::ParallelCsrMatrixClass( ParallelCsrMatrixClass<complexs> &&mat);
   template ParallelCsrMatrixClass<complexd>::ParallelCsrMatrixClass( ParallelCsrMatrixClass<complexd> &&mat);
   
   template <typename T>
   ParallelCsrMatrixClass<T>& ParallelCsrMatrixClass<T>::operator= (const ParallelCsrMatrixClass<T> &mat)
   {
      this->Clear();
      ParallelLogClass::operator=(mat);
      this->_nrow_global = mat._nrow_global;
      this->_ncol_global = mat._ncol_global;
      this->_nrow_start = mat._nrow_start;
      this->_ncol_start = mat._ncol_start;
      this->_nrow_local = mat._nrow_local;
      this->_ncol_local = mat._ncol_local;
      this->_is_offd_map_sorted = mat._is_offd_map_sorted;
      this->_diag_mat = mat._diag_mat;
      this->_offd_mat = mat._offd_mat;
      this->_offd_map_v = mat._offd_map_v;
      this->_matvec_working_vec = mat._matvec_working_vec;
      this->_trans_matvec_working_vec = mat._trans_matvec_working_vec;
      this->_comm_helper = mat._comm_helper;
      this->_trans_comm_helper = mat._trans_comm_helper;
      this->_separator_ndom = mat._separator_ndom;
      this->_separator_domi = mat._separator_domi;
      return *this;
   }
   template ParallelCsrMatrixClass<float>& ParallelCsrMatrixClass<float>::operator= (const ParallelCsrMatrixClass<float> &mat);
   template ParallelCsrMatrixClass<double>& ParallelCsrMatrixClass<double>::operator= (const ParallelCsrMatrixClass<double> &mat);
   template ParallelCsrMatrixClass<complexs>& ParallelCsrMatrixClass<complexs>::operator= (const ParallelCsrMatrixClass<complexs> &mat);
   template ParallelCsrMatrixClass<complexd>& ParallelCsrMatrixClass<complexd>::operator= (const ParallelCsrMatrixClass<complexd> &mat);
   
   template <typename T>
   ParallelCsrMatrixClass<T>& ParallelCsrMatrixClass<T>::operator= ( ParallelCsrMatrixClass<T> &&mat)
   {
      this->Clear();
      ParallelLogClass::operator=(std::move(mat));
      this->_nrow_global = mat._nrow_global;
      mat._nrow_global = 0;
      this->_ncol_global = mat._ncol_global;
      mat._ncol_global = 0;
      this->_nrow_start = mat._nrow_start;
      mat._nrow_start = 0;
      this->_ncol_start = mat._ncol_start;
      mat._ncol_start = 0;
      this->_nrow_local = mat._nrow_local;
      mat._nrow_local = 0;
      this->_ncol_local = mat._ncol_local;
      mat._ncol_local = 0;
      this->_is_offd_map_sorted = mat._is_offd_map_sorted;
      mat._is_offd_map_sorted = false;
      this->_diag_mat = std::move(mat._diag_mat);
      this->_offd_mat = std::move(mat._offd_mat);
      this->_offd_map_v = std::move(mat._offd_map_v);
      this->_matvec_working_vec = std::move(mat._matvec_working_vec);
      this->_trans_matvec_working_vec = std::move(mat._trans_matvec_working_vec);
      this->_comm_helper = std::move(mat._comm_helper);
      this->_trans_comm_helper = std::move(mat._trans_comm_helper);
      this->_separator_ndom = mat._separator_ndom;
      mat._separator_ndom = 0;
      this->_separator_domi = std::move(mat._separator_domi);
      return *this;
   }
   template ParallelCsrMatrixClass<float>& ParallelCsrMatrixClass<float>::operator= ( ParallelCsrMatrixClass<float> &&mat);
   template ParallelCsrMatrixClass<double>& ParallelCsrMatrixClass<double>::operator= ( ParallelCsrMatrixClass<double> &&mat);
   template ParallelCsrMatrixClass<complexs>& ParallelCsrMatrixClass<complexs>::operator= ( ParallelCsrMatrixClass<complexs> &&mat);
   template ParallelCsrMatrixClass<complexd>& ParallelCsrMatrixClass<complexd>::operator= ( ParallelCsrMatrixClass<complexd> &&mat);
   
   template <typename T>
   ParallelCsrMatrixClass<T>::~ParallelCsrMatrixClass()
   {
      this->Clear();
   }
   template ParallelCsrMatrixClass<float>::~ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<double>::~ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<complexs>::~ParallelCsrMatrixClass();
   template ParallelCsrMatrixClass<complexd>::~ParallelCsrMatrixClass();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Clear()
   {
      /* base class clear */
      MatrixClass<T>::Clear();
      
      _nrow_global = 0;
      _ncol_global = 0;
      _nrow_start = 0;
      _ncol_start = 0;
      _nrow_local = 0;
      _ncol_local = 0;
      _is_offd_map_sorted = false;
      _diag_mat.Clear();
      _offd_mat.Clear();
      _offd_map_v.Clear();
      _comm_helper.Clear();
      _trans_comm_helper.Clear();
      _matvec_working_vec.Clear();
      _trans_matvec_working_vec.Clear();
      
      _separator_domi.Clear();
      _separator_ndom = 0;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Clear();
   template int ParallelCsrMatrixClass<double>::Clear();
   template int ParallelCsrMatrixClass<complexs>::Clear();
   template int ParallelCsrMatrixClass<complexd>::Clear();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Setup(int nrow_local, int ncol_local, parallel_log &parlog)
   {
      /* setup the parallel matrix */
      long int nrow_start, ncol_start, nrow_global, ncol_global;
      
      MPI_Comm comm;
      int      np, myid;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      PargemslrNLocalToNGlobal( nrow_local, ncol_local, nrow_start, ncol_start, nrow_global, ncol_global, comm);
      
      return this->Setup( nrow_local, nrow_start, nrow_global, ncol_local, ncol_start, ncol_global, parlog);
   }
   template int ParallelCsrMatrixClass<float>::Setup(int nrow_local, int ncol_local, parallel_log &parlog);
   template int ParallelCsrMatrixClass<double>::Setup(int nrow_local, int ncol_local, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexs>::Setup(int nrow_local, int ncol_local, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::Setup(int nrow_local, int ncol_local, parallel_log &parlog);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog)
   {
      this->Clear();
      
      /* information */
      this->_nrow_local    = nrow_local;
      this->_nrow_global   = nrow_global;
      this->_nrow_start    = nrow_start;
      this->_ncol_local    = ncol_local;
      this->_ncol_global   = ncol_global;
      this->_ncol_start    = ncol_start;
      
      /* parallel info */
      MPI_Comm comm;
      int np, myid;
      parlog.GetMpiInfo(np, myid, comm);
      
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup( comm, this->_comm));
      this->_size = np;
      this->_rank = myid;
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog);
   template int ParallelCsrMatrixClass<double>::Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexs>::Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::GetDataLocation() const
   {
      PARGEMSLR_CHKERR( this->_diag_mat.GetDataLocation() != this->_offd_mat.GetDataLocation() );
      return this->_diag_mat.GetDataLocation();
   }
   template int ParallelCsrMatrixClass<float>::GetDataLocation() const;
   template int ParallelCsrMatrixClass<double>::GetDataLocation() const;
   template int ParallelCsrMatrixClass<complexs>::GetDataLocation() const;
   template int ParallelCsrMatrixClass<complexd>::GetDataLocation() const;
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::GetNumRowsLocal() const
   {
      return this->_nrow_local;
   }
   template int ParallelCsrMatrixClass<float>::GetNumRowsLocal() const;
   template int ParallelCsrMatrixClass<double>::GetNumRowsLocal() const;
   template int ParallelCsrMatrixClass<complexs>::GetNumRowsLocal() const;
   template int ParallelCsrMatrixClass<complexd>::GetNumRowsLocal() const;
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::GetNumColsLocal() const
   {
      return this->_ncol_local;
   }
   template int ParallelCsrMatrixClass<float>::GetNumColsLocal() const;
   template int ParallelCsrMatrixClass<double>::GetNumColsLocal() const;
   template int ParallelCsrMatrixClass<complexs>::GetNumColsLocal() const;
   template int ParallelCsrMatrixClass<complexd>::GetNumColsLocal() const;
   
   template <typename T>
   long int ParallelCsrMatrixClass<T>::GetNumRowsGlobal() const
   {
      return this->_nrow_global;
   }
   template long int ParallelCsrMatrixClass<float>::GetNumRowsGlobal() const;
   template long int ParallelCsrMatrixClass<double>::GetNumRowsGlobal() const;
   template long int ParallelCsrMatrixClass<complexs>::GetNumRowsGlobal() const;
   template long int ParallelCsrMatrixClass<complexd>::GetNumRowsGlobal() const;
   
   template <typename T>
   long int ParallelCsrMatrixClass<T>::GetNumColsGlobal() const
   {
      return this->_ncol_global;
   }
   template long int ParallelCsrMatrixClass<float>::GetNumColsGlobal() const;
   template long int ParallelCsrMatrixClass<double>::GetNumColsGlobal() const;
   template long int ParallelCsrMatrixClass<complexs>::GetNumColsGlobal() const;
   template long int ParallelCsrMatrixClass<complexd>::GetNumColsGlobal() const;
   
   template <typename T>
   long int ParallelCsrMatrixClass<T>::GetRowStartGlobal() const
   {
      return this->_nrow_start;
   }
   template long int ParallelCsrMatrixClass<float>::GetRowStartGlobal() const;
   template long int ParallelCsrMatrixClass<double>::GetRowStartGlobal() const;
   template long int ParallelCsrMatrixClass<complexs>::GetRowStartGlobal() const;
   template long int ParallelCsrMatrixClass<complexd>::GetRowStartGlobal() const;
   
   template <typename T>
   long int ParallelCsrMatrixClass<T>::GetColStartGlobal() const
   {
      return this->_ncol_start;
   }
   template long int ParallelCsrMatrixClass<float>::GetColStartGlobal() const;
   template long int ParallelCsrMatrixClass<double>::GetColStartGlobal() const;
   template long int ParallelCsrMatrixClass<complexs>::GetColStartGlobal() const;
   template long int ParallelCsrMatrixClass<complexd>::GetColStartGlobal() const;
   
   template <typename T>
   long int ParallelCsrMatrixClass<T>::GetNumNonzeros() const
   {
      long int nnz_global = 0;
      
      if(this->_comm)
      {
         long int nnz_local;
         nnz_local = this->_diag_mat.GetNumNonzeros() + this->_offd_mat.GetNumNonzeros();
         PargemslrMpiAllreduce( &nnz_local, &nnz_global, 1, MPI_SUM, *(this->_comm));
      }
      
      return nnz_global;
   }
   template long int ParallelCsrMatrixClass<float>::GetNumNonzeros() const;
   template long int ParallelCsrMatrixClass<double>::GetNumNonzeros() const;
   template long int ParallelCsrMatrixClass<complexs>::GetNumNonzeros() const;
   template long int ParallelCsrMatrixClass<complexd>::GetNumNonzeros() const;
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy)
   {
      int i, j, j1, j2;
      int np, myid;
      MPI_Comm comm;
      
      int *diag_i = this->_diag_mat.GetI();
      int *offd_i = this->_offd_mat.GetI();
      int *diag_j = this->_diag_mat.GetJ();
      int *offd_j = this->_offd_mat.GetJ();
      
      this->GetMpiInfo(np, myid, comm);
      
      vtxdist.Setup(np+1);
      vtxdist[np] = this->_nrow_global;

      PARGEMSLR_MPI_CALL( PargemslrMpiAllgather( &this->_nrow_start, 1, vtxdist.GetData(), comm) );
      
      /* diagonal removed */
      xadj.Setup(this->_nrow_local+1);
      adjncy.Setup(this->_diag_mat.GetNumNonzeros()+this->_offd_mat.GetNumNonzeros()-this->_nrow_local);
      
      xadj[0] = 0;
      for (i = 0; i < this->_nrow_local; i++)
      {
         xadj[i+1] = xadj[i];
         j1 = diag_i[i];
         j2 = diag_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            if(diag_j[j] != i)
            {
               /* only add non-diagonal entry */
               adjncy[xadj[i+1]] = diag_j[j] + this->_nrow_start;
               xadj[i+1]++;
            }
         }
         
         j1 = offd_i[i];
         j2 = offd_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            adjncy[xadj[i+1]] = this->_offd_map_v[offd_j[j]];
            xadj[i+1]++;
         }
      }
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy);
   template int ParallelCsrMatrixClass<double>::GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy);
   template int ParallelCsrMatrixClass<complexs>::GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy);
   template int ParallelCsrMatrixClass<complexd>::GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy);
   
   template <typename T>
   CsrMatrixClass<T>& ParallelCsrMatrixClass<T>::GetDiagMat()
   {
      return this->_diag_mat;
   }
   template CsrMatrixClass<float>& ParallelCsrMatrixClass<float>::GetDiagMat();
   template CsrMatrixClass<double>& ParallelCsrMatrixClass<double>::GetDiagMat();
   template CsrMatrixClass<complexs>& ParallelCsrMatrixClass<complexs>::GetDiagMat();
   template CsrMatrixClass<complexd>& ParallelCsrMatrixClass<complexd>::GetDiagMat();
   
   template <typename T>
   CsrMatrixClass<T>& ParallelCsrMatrixClass<T>::GetOffdMat()
   {
      return this->_offd_mat;
   }
   template CsrMatrixClass<float>& ParallelCsrMatrixClass<float>::GetOffdMat();
   template CsrMatrixClass<double>& ParallelCsrMatrixClass<double>::GetOffdMat();
   template CsrMatrixClass<complexs>& ParallelCsrMatrixClass<complexs>::GetOffdMat();
   template CsrMatrixClass<complexd>& ParallelCsrMatrixClass<complexd>::GetOffdMat();
   
   template <typename T>
   vector_long& ParallelCsrMatrixClass<T>::GetOffdMap()
   {
      return this->_offd_map_v;
   }
   template vector_long& ParallelCsrMatrixClass<float>::GetOffdMap();
   template vector_long& ParallelCsrMatrixClass<double>::GetOffdMap();
   template vector_long& ParallelCsrMatrixClass<complexs>::GetOffdMap();
   template vector_long& ParallelCsrMatrixClass<complexd>::GetOffdMap();
   
   template <typename T>
   bool& ParallelCsrMatrixClass<T>::IsOffdMapSorted()
   {
      return this->_is_offd_map_sorted;
   }
   template bool& ParallelCsrMatrixClass<float>::IsOffdMapSorted();
   template bool& ParallelCsrMatrixClass<double>::IsOffdMapSorted();
   template bool& ParallelCsrMatrixClass<complexs>::IsOffdMapSorted();
   template bool& ParallelCsrMatrixClass<complexd>::IsOffdMapSorted();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SetOffdMatSorted(bool is_offd_map_sorted)
   {
      this->_is_offd_map_sorted = is_offd_map_sorted;
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::SetOffdMatSorted(bool is_offd_map_sorted);
   template int ParallelCsrMatrixClass<double>::SetOffdMatSorted(bool is_offd_map_sorted);
   template int ParallelCsrMatrixClass<complexs>::SetOffdMatSorted(bool is_offd_map_sorted);
   template int ParallelCsrMatrixClass<complexd>::SetOffdMatSorted(bool is_offd_map_sorted);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SortOffdMap()
   {
      int           i, j, j1, j2, noffd, nrows, location;
      
      noffd = this->_offd_map_v.GetLengthLocal();
      
      if(this->_is_offd_map_sorted || noffd == 0)
      {
         /* we know that it's already sorted */
         return PARGEMSLR_SUCCESS;
      }
      
      location = this->GetDataLocation();
      
      if(location == kMemoryDevice)
      {
         //PARGEMSLR_ERROR("Parallel CSR sort off-diagonal matrix only works on the host memory. (This function is used when setting up the matvec.)");
         //return PARGEMSLR_ERROR_MEMORY_LOCATION;
         PARGEMSLR_WARNING("Parallel CSR sort off-diagonal matrix only works on the host memory, moving data to the host. (This function is used when setting up the matvec.)");
         this->MoveData(kMemoryHost);
      }
      
      int            *A_i, *A_j;
      bool           sorted;
      vector_int     order, iorder;
      
      /* we don't know if its already sorted, but we can avoid quick sort if it's indeed sorted */
      sorted = true;
      for(i = 1 ; i < noffd ; i ++)
      {
         if(this->_offd_map_v[i-1] > this->_offd_map_v[i])
         {
            sorted = false;
            break;
         }
      }
      if(sorted)
      {
         this->_is_offd_map_sorted = true;
         return PARGEMSLR_SUCCESS;
      }
      
      /* sort in ascending order, note that the vector has not been modified yet */
      this->_offd_map_v.Sort( order, true, false);
      
      /* modify the offd_map */
      this->_offd_map_v.Perm(order);
      
      /* now apply the permutation */
      iorder.Setup( noffd, false);
      for(i = 0 ; i < noffd ; i ++)
      {
         iorder[order[i]] = i;
      }
      
      A_i = this->_offd_mat.GetI();
      A_j = this->_offd_mat.GetJ();
      
      nrows = this->_nrow_local;
      
      for(i = 0 ; i < nrows ; i ++)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            A_j[j] = iorder[A_j[j]];
         }
      }
      
      /* De allocation */
      
      order.Clear();
      iorder.Clear();
      
      this->_is_offd_map_sorted = true;
      
      /* move back to device when necessary */
      this->MoveData(location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::SortOffdMap();
   template int ParallelCsrMatrixClass<double>::SortOffdMap();
   template int ParallelCsrMatrixClass<complexs>::SortOffdMap();
   template int ParallelCsrMatrixClass<complexd>::SortOffdMap();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SetupVectorPtrStr(ParallelVectorClass<T> &vec)
   {
      return vec.SetupPtrStr(*this);
   }
   template int ParallelCsrMatrixClass<float>::SetupVectorPtrStr(ParallelVectorClass<float> &vec);
   template int ParallelCsrMatrixClass<double>::SetupVectorPtrStr(ParallelVectorClass<double> &vec);
   template int ParallelCsrMatrixClass<complexs>::SetupVectorPtrStr(ParallelVectorClass<complexs> &vec);
   template int ParallelCsrMatrixClass<complexd>::SetupVectorPtrStr(ParallelVectorClass<complexd> &vec);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Eye()
   {
      /* don't need to worry for the host version */
      this->_diag_mat.Eye();
      this->_offd_mat.Setup(this->_nrow_local, 0, 0);
      
      _is_offd_map_sorted = true;
      _offd_map_v.Clear();
      _comm_helper.Clear();
      _trans_comm_helper.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::Eye();
   template int ParallelCsrMatrixClass<double>::Eye();
   template int ParallelCsrMatrixClass<complexs>::Eye();
   template int ParallelCsrMatrixClass<complexd>::Eye();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::MoveData( const int &location)
   {
      this->_diag_mat.MoveData(location);
      this->_offd_mat.MoveData(location);
      
      this->_matvec_working_vec.MoveData(location);
      this->_trans_matvec_working_vec.MoveData(location);
      
      if(this->_comm_helper._is_ready)
      {
         this->_comm_helper.MoveData(location);
      }
      if(this->_trans_comm_helper._is_ready)
      {
         this->_trans_comm_helper.MoveData(location);
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::MoveData( const int &location);
   template int ParallelCsrMatrixClass<double>::MoveData( const int &location);
   template int ParallelCsrMatrixClass<complexs>::MoveData( const int &location);
   template int ParallelCsrMatrixClass<complexd>::MoveData( const int &location);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Fill(const T &v)
   {
      this->_diag_mat.Fill(v);
      this->_offd_mat.Fill(v);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Fill(const float &v);
   template int ParallelCsrMatrixClass<double>::Fill(const double &v);
   template int ParallelCsrMatrixClass<complexs>::Fill(const complexs &v);
   template int ParallelCsrMatrixClass<complexd>::Fill(const complexd &v);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Scale(const T &alpha)
   {
      
      this->_diag_mat.Scale(alpha);
      this->_offd_mat.Scale(alpha);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Scale(const float &alpha);
   template int ParallelCsrMatrixClass<double>::Scale(const double &alpha);
   template int ParallelCsrMatrixClass<complexs>::Scale(const complexs &alpha);
   template int ParallelCsrMatrixClass<complexd>::Scale(const complexd &alpha);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::GetDiagScale(T &scale)
   {
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      T local_scale;
      RealDataType local_scale_real, scale_real;
      MPI_Comm comm;
      int np, myid;
      this->GetMpiInfo(np, myid, comm);
      this->_diag_mat.GetDiagScale(local_scale);
      local_scale_real = PargemslrAbs(local_scale);
      PARGEMSLR_MPI_CALL( PargemslrMpiAllreduce( &local_scale_real, &scale_real, 1, MPI_MAX, comm) );
      scale = T(scale_real);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::GetDiagScale(float &scale);
   template int ParallelCsrMatrixClass<double>::GetDiagScale(double &scale);
   template int ParallelCsrMatrixClass<complexs>::GetDiagScale(complexs &scale);
   template int ParallelCsrMatrixClass<complexd>::GetDiagScale(complexd &scale);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SetupMatvec()
   {
      
      this->SetupMatvecStart();
      this->SetupMatvecOver();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::SetupMatvec();
   template int ParallelCsrMatrixClass<double>::SetupMatvec();
   template int ParallelCsrMatrixClass<complexs>::SetupMatvec();
   template int ParallelCsrMatrixClass<complexd>::SetupMatvec();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SetupMatvecStart()
   {
      
      if(this->_comm_helper._is_ready || this->_comm_helper._is_waiting)
      {
         /* do nothing if this function is already called */
         return PARGEMSLR_SUCCESS;
      }
      
      //PARGEMSLR_CHKERR(this->_comm_helper._is_waiting);
      
      /*------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       *------------------------*/
      
      int                 i, j, k, noffd, startidx, endidx, sendsize, recvsize;
      int                 upid, downid, sends, recvs, reqidx, sendrecv, sendidx, recvidx;
      vector_long         send_buff_long_v, recv_buff_long_v, offd_map_vec; 
      
      /* mpi info and copy the current MPI_Comm for the parallel matrix */
      MPI_Comm comm;
      int      myid, np;
      
      this->GetMpiInfo(np, myid, comm);
      
      this->_comm_helper._commref = comm;
      this->_comm_helper._rank = myid;
      this->_comm_helper._size = np;
      
      this->_comm_helper.GetMpiInfo(np, myid, comm);
      
      /*------------------------
       * 2: Build data structures
       *------------------------*/
      
      this->_comm_helper._n_in = this->_diag_mat.GetNumColsLocal();
      this->_comm_helper._n_out = this->_offd_mat.GetNumColsLocal();
      
      /* sort offdiagonal matrix */
      this->SortOffdMap();
      
      noffd = this->_offd_map_v.GetLengthLocal();
      
      /* If the neiborhoods of each MPI rank is known, we can simplify the communication */
      if(this->_comm_helper._is_nbhd_built)
      {
         /* nbhd is known, only communicate with them */
         
         /*------------------------------------------------
          * 2.1: Sent the local column start/end to nbhds
          * The vector of the matvec has same distribution
          * as the columns of A. 
          *------------------------------------------------*/
          
         /* prepare the data */
         sends = this->_comm_helper._send_to_v.GetLengthLocal();
         this->_comm_helper._send_idx_v2.resize(sends);
         recvs = this->_comm_helper._recv_from_v.GetLengthLocal();
         this->_comm_helper._recv_idx_v2.resize(recvs);
         this->_comm_helper._idx_helper_v2.resize(recvs);
         
         sendrecv = sends + recvs;
         this->_comm_helper._requests_v.resize(sendrecv);
         send_buff_long_v.Setup(2, false);
         recv_buff_long_v.Setup(2*PargemslrMax(sends,recvs), false);
         send_buff_long_v[0] = _ncol_start;
         send_buff_long_v[1] = _ncol_start + _ncol_local;
         
         /* do the communication */
         
         reqidx = 0;
         for(i = 0 ; i < sends ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_buff_long_v.GetData(), 2, this->_comm_helper._send_to_v[i], 
                              this->_comm_helper._send_to_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
            
         }
         
         for(i = 0 ; i < recvs ; i ++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_buff_long_v.GetData()+2*i, 2, this->_comm_helper._recv_from_v[i],
                              myid, comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         }
         
         /*------------------------------------------------
          * 2.2: Determine the indices of external vectors.
          * Setup the recv_idx array.
          *------------------------------------------------*/
         
         for(i = 0 ; i < sendrecv ; i ++)
         {
            PARGEMSLR_MPI_CALL( MPI_Waitany( sendrecv, this->_comm_helper._requests_v.data(), &reqidx, MPI_STATUSES_IGNORE) );
            if(reqidx >= sends)
            {
               reqidx -= sends;
               /* apply binary search ot check if we have those data */
               this->_offd_map_v.BinarySearch( recv_buff_long_v[2*reqidx], startidx, true);
               
               endidx = startidx;
               while(endidx < noffd && this->_offd_map_v[endidx] < recv_buff_long_v[2*reqidx+1])
               {
                  endidx++;
               }
               
               /* get recv indices */
               recvsize = endidx - startidx;
               
               this->_comm_helper._recv_idx_v2[reqidx].Setup(recvsize);
               this->_comm_helper._idx_helper_v2[reqidx].Setup(recvsize);
               
               for(j = startidx, k = 0 ; j < endidx ; j ++ , k ++)
               {
                  this->_comm_helper._recv_idx_v2[reqidx][k] = j;
                  this->_comm_helper._idx_helper_v2[reqidx][k] = (int)(this->_offd_map_v[j]-recv_buff_long_v[2*reqidx]);
               }
            }
         }
         
         /*-----------------------------------------------
          * 2.3: Send the size of the request data to nbhds
          *-----------------------------------------------*/
         
         reqidx=0;
         j = 0;
         for(i = 0 ; i < sends ; i++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( recv_buff_long_v.GetData()+2*i, 1, this->_comm_helper._send_to_v[i],
                              myid*this->_comm_helper._send_to_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         } 
         if(j < recvs)
         {
            send_buff_long_v[0] = this->_comm_helper._idx_helper_v2[j].GetLengthLocal();
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_buff_long_v.GetData(), 1, this->_comm_helper._recv_from_v[j], 
                              myid*this->_comm_helper._recv_from_v[j], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
            j++;
         }
         
         for(i = 0 ; i < sendrecv ; i ++)
         {
            PARGEMSLR_MPI_CALL( MPI_Waitany( sends + j, this->_comm_helper._requests_v.data(), &reqidx, MPI_STATUSES_IGNORE) );
            if(reqidx < sends)
            {
               this->_comm_helper._send_idx_v2[reqidx].Setup(recv_buff_long_v[reqidx*2]);
            }
            else
            {
               if(j < recvs)
               {
                  send_buff_long_v[0] = this->_comm_helper._idx_helper_v2[j].GetLengthLocal();
                  PARGEMSLR_MPI_CALL( PargemslrMpiIsend( send_buff_long_v.GetData(), 1, this->_comm_helper._recv_from_v[j], 
                              myid*this->_comm_helper._recv_from_v[j], comm, &(this->_comm_helper._requests_v[sends+j]) ) );
                  j++;
               }
            }
         }
         
         /*-----------------------------------------------
          * 2.4: Send indices of external vector to their holders.
          * Setup the send_idx array.
          *-----------------------------------------------*/
         
         reqidx=0;
         for(i = 0 ; i < recvs ; i++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( this->_comm_helper._idx_helper_v2[i].GetData(), this->_comm_helper._idx_helper_v2[i].GetLengthLocal(), 
                              this->_comm_helper._recv_from_v[i], myid*this->_comm_helper._recv_from_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         }
         for(i = 0 ; i < sends ; i++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( this->_comm_helper._send_idx_v2[i].GetData(), this->_comm_helper._send_idx_v2[i].GetLengthLocal(), 
                              this->_comm_helper._send_to_v[i], myid*this->_comm_helper._send_to_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         }
         
      }
      else
      {
         /* nbhd is not known, need to find nbhds */
         
         /*------------------------------------------------
          * 2.1: We don't know the nbhds information,
          * need to communicate with all other MPI ranks.
          * Starting from 1, each time add shift by 1.
          * Sent the local column start/end to myid + shift,
          * get the local column start/end from myid - shift.
          * If myid requires information from myid + shift,
          * build the corresponding data structure.
          *------------------------------------------------*/
         
         this->_comm_helper._requests_v.resize(2);
         send_buff_long_v.Setup(2, false);
         recv_buff_long_v.Setup(2, false);
         sendidx = 0;
         recvidx = 0;
         this->_comm_helper._recv_from_v.Setup(np);
         this->_comm_helper._recv_idx_v2.resize(np);
         this->_comm_helper._idx_helper_v2.resize(np);
         this->_comm_helper._send_to_v.Setup(np);
         this->_comm_helper._send_idx_v2.resize(np);
         
         for(i = 1 ; i < np ; i ++)
         {
            upid = (myid + i) % np;
            downid = (myid - i + np) % np;
            
            send_buff_long_v[0] = _ncol_start;
            send_buff_long_v[1] = _ncol_start + _ncol_local;
            
            /* send range to up, and get from down 
             * TODO: update this function to avoid the direct use of MPI_LONG
             */
            PARGEMSLR_MPI_CALL( MPI_Sendrecv( send_buff_long_v.GetData(), 2, MPI_LONG, upid, myid*upid,
                              recv_buff_long_v.GetData(), 2, MPI_LONG, downid, myid*downid,
                              comm, MPI_STATUS_IGNORE) );
            //MPI_Isend(send_buff_long_v._data, 2, GEMSLR_MPI_LONG, upid, myid*upid, *(matvec_helper.mpi_comm), &(matvec_helper.requests[0]));
            //MPI_Irecv(recv_buff_long_v._data, 2, GEMSLR_MPI_LONG, downid, myid*downid, *(matvec_helper.mpi_comm), &(matvec_helper.requests[1]));
            
            /* wait for communication to end */
            //MPI_Waitall( 2, matvec_helper.requests.data(), MPI_STATUSES_IGNORE);
            
            /* apply binary search ot check if we have those data */
            this->_offd_map_v.BinarySearch( recv_buff_long_v[0], startidx, true);
            
            endidx = startidx;
            while(endidx < noffd && this->_offd_map_v[endidx] < recv_buff_long_v[1])
            {
               endidx++;
            }
            
            /* check if we have data */
            recvsize = endidx - startidx;
            
            /* send range to up */
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend(&recvsize, 1, downid, myid*downid, comm, &(this->_comm_helper._requests_v[0])) );
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv(&sendsize, 1, upid, myid*upid, comm, &(this->_comm_helper._requests_v[1])) );
            if(recvsize > 0)
            {
               this->_comm_helper._recv_from_v[recvidx] = downid;
               this->_comm_helper._recv_idx_v2[recvidx].Setup(recvsize);
               this->_comm_helper._idx_helper_v2[recvidx].Setup(recvsize);
               
               for(j = startidx, k = 0 ; j < endidx ; j ++, k ++)
               {
                  this->_comm_helper._recv_idx_v2[recvidx][k] = j;
                  this->_comm_helper._idx_helper_v2[recvidx][k] = (int)(this->_offd_map_v[j]-recv_buff_long_v[0]);
               }
               recvidx++;
            }
            PARGEMSLR_MPI_CALL( MPI_Wait(this->_comm_helper._requests_v.data()+1, MPI_STATUSES_IGNORE) );
            if(sendsize > 0)
            {
               this->_comm_helper._send_to_v[sendidx] = upid;
               this->_comm_helper._send_idx_v2[sendidx].Setup(sendsize);
               sendidx++;
            }
            PARGEMSLR_MPI_CALL( MPI_Wait(this->_comm_helper._requests_v.data(), MPI_STATUSES_IGNORE) );
         }
         this->_comm_helper._recv_from_v.Resize(recvidx, true, true);
         this->_comm_helper._recv_idx_v2.resize(recvidx);
         this->_comm_helper._idx_helper_v2.resize(recvidx);
         this->_comm_helper._send_to_v.Resize(sendidx, true, true);
         this->_comm_helper._send_idx_v2.resize(sendidx);
         
         /*-----------------------------------------------
          * 2.2: Send indices of external vector to their holders.
          * Setup the send_idx array.
          *-----------------------------------------------*/
         
         recvs = this->_comm_helper._recv_from_v.GetLengthLocal();
         sends = this->_comm_helper._send_to_v.GetLengthLocal();
         this->_comm_helper._requests_v.resize(recvs+sends);
         
         reqidx=0;
         for(i = 0 ; i < recvs ; i++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIsend( this->_comm_helper._idx_helper_v2[i].GetData(), this->_comm_helper._idx_helper_v2[i].GetLengthLocal(), 
                              this->_comm_helper._recv_from_v[i], myid*this->_comm_helper._recv_from_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         }
         
         for(i = 0 ; i < sends ; i++)
         {
            PARGEMSLR_MPI_CALL( PargemslrMpiIrecv( this->_comm_helper._send_idx_v2[i].GetData(), this->_comm_helper._send_idx_v2[i].GetLengthLocal(), 
                              this->_comm_helper._send_to_v[i], myid*this->_comm_helper._send_to_v[i], comm, &(this->_comm_helper._requests_v[reqidx++]) ) );
         }
         
      }
      
      /* now create buffer, the default buffer size is used to transfer type T */
      int location = this->GetDataLocation();
      this->_comm_helper.CreateHostBuffer(sizeof(T));
#ifdef PARGEMSLR_CUDA
      if(location == kMemoryDevice)
      {
         this->_comm_helper.CreateDeviceBuffer(sizeof(T));
      }
#endif
      
      this->_matvec_working_vec.Setup( this->_offd_map_v.GetLengthLocal(), location, true);
      
      /* TODO: setup working vector for matvec */
      this->_comm_helper._is_waiting = true;
      
      send_buff_long_v.Clear();
      recv_buff_long_v.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::SetupMatvecStart();
   template int ParallelCsrMatrixClass<double>::SetupMatvecStart();
   template int ParallelCsrMatrixClass<complexs>::SetupMatvecStart();
   template int ParallelCsrMatrixClass<complexd>::SetupMatvecStart();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SetupMatvecOver()
   {
      if(this->_comm_helper._is_ready == true)
      {
         /* do nothing if the helper is ready */
         return PARGEMSLR_SUCCESS;
      }
      PARGEMSLR_CHKERR( !(this->_comm_helper._is_waiting) ); 
      
      int      i, sends, recvs, send_plus_recv, idx;
      
      MPI_Comm comm;
      int      myid, np;
      
      this->_comm_helper.GetMpiInfo(np, myid, comm);
      
      sends          = this->_comm_helper._send_to_v.GetLengthLocal();
      recvs          = this->_comm_helper._recv_from_v.GetLengthLocal();
      send_plus_recv = sends+ recvs;
      
      /* wait till communication finished */
      for(i = 0 ; i < send_plus_recv ; i ++)
      {
         PARGEMSLR_MPI_CALL( MPI_Waitany(send_plus_recv, this->_comm_helper._requests_v.data(), &idx, MPI_STATUSES_IGNORE) );
         if(idx < recvs)
         {
            //this is a send process, clear send buffer
            this->_comm_helper._idx_helper_v2[idx].Clear();
         }
      }
      
      /* now ready to be used, move to assigned memory location */
      this->_comm_helper.MoveData(this->GetDataLocation());
      this->_comm_helper._is_waiting = false;
      this->_comm_helper._is_ready = true;
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::SetupMatvecOver();
   template int ParallelCsrMatrixClass<double>::SetupMatvecOver();
   template int ParallelCsrMatrixClass<complexs>::SetupMatvecOver();
   template int ParallelCsrMatrixClass<complexd>::SetupMatvecOver();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      T one = 1.0;
      
      if(trans == 'N')
      {
         if( !(this->_comm_helper._is_ready) )
         {
            this->SetupMatvec();
         }
         
         /* start communication of offd part */
         this->_comm_helper.DataTransferStart(x, x.GetDataLocation());
         
         /* apply diagonal matvec while communicating */
         this->_diag_mat.MatVec( 'N', alpha, x, beta, y);
         
         /* finishing up the communication */
         this->_comm_helper.DataTransferOver(this->_matvec_working_vec, this->_matvec_working_vec.GetDataLocation());
         
         /* now apply offd matvec */
         if(this->_matvec_working_vec.GetLengthLocal() > 0)
         {
            this->_offd_mat.MatVec( 'N', alpha, this->_matvec_working_vec, one, y);
         }
         
         /* done */
         
      }
      else
      {
         PARGEMSLR_ERROR("parallel CSR matrix transport matvec not supported yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int ParallelCsrMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int ParallelCsrMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int ParallelCsrMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z)
   {
      T one = 1.0;
      T zero = T(0.0);
      
      z.Fill(zero);
      z.Axpy(one, y);
      
      if(trans == 'N')
      {
         if( !(this->_comm_helper._is_ready) )
         {
            this->SetupMatvec();
         }
         
         /* start communication of offd part */
         this->_comm_helper.DataTransferStart(x, x.GetDataLocation());
         
         /* apply diagonal matvec while communicating */
         this->_diag_mat.MatVec( 'N', alpha, x, beta, z);
         
         /* finishing up the communication */
         this->_comm_helper.DataTransferOver(this->_matvec_working_vec, this->_matvec_working_vec.GetDataLocation());
         
         /* now apply offd matvec */
         if(this->_matvec_working_vec.GetLengthLocal() > 0)
         {
            this->_offd_mat.MatVec( 'N', alpha, this->_matvec_working_vec, one, z);
         }
         
         /* done */
         
      }
      else
      {
         PARGEMSLR_ERROR("parallel CSR matrix transport matvec not supported yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::MatVec( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, const VectorClass<float> &y, VectorClass<float> &z);
   template int ParallelCsrMatrixClass<double>::MatVec( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, const VectorClass<double> &y, VectorClass<double> &z);
   template int ParallelCsrMatrixClass<complexs>::MatVec( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, const VectorClass<complexs> &y, VectorClass<complexs> &z);
   template int ParallelCsrMatrixClass<complexd>::MatVec( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, const VectorClass<complexd> &y, VectorClass<complexd> &z);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::MatVecOffd( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y)
   {
      if(trans == 'N')
      {
         if( !(this->_comm_helper._is_ready) )
         {
            this->SetupMatvec();
         }
         
         /* start communication of offd part */
         this->_comm_helper.DataTransferStart(x, x.GetDataLocation());
         
         /* finishing up the communication */
         this->_comm_helper.DataTransferOver(this->_matvec_working_vec, this->_matvec_working_vec.GetDataLocation());
         
         /* now apply offd matvec */
         if(this->_matvec_working_vec.GetLengthLocal() > 0)
         {
            this->_offd_mat.MatVec( 'N', alpha, this->_matvec_working_vec, beta, y);
         }
         
         /* done */
         
      }
      else
      {
         PARGEMSLR_ERROR("parallel CSR matrix transport matvec not supported yet.");
         return PARGEMSLR_ERROR_FUNCTION_CALL_ERR;
      }
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::MatVecOffd( char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y);
   template int ParallelCsrMatrixClass<double>::MatVecOffd( char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y);
   template int ParallelCsrMatrixClass<complexs>::MatVecOffd( char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y);
   template int ParallelCsrMatrixClass<complexd>::MatVecOffd( char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::MatMat( const T &alpha, const ParallelCsrMatrixClass<T> &A, char transa, const ParallelCsrMatrixClass<T> &B, char transb, const T &beta)
   {
      PARGEMSLR_ERROR("parallel CSR matrix matmat not supported yet.");
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::MatMat( const float &alpha, const ParallelCsrMatrixClass<float> &A, char transa, const ParallelCsrMatrixClass<float> &B, char transb, const float &beta);
   template int ParallelCsrMatrixClass<double>::MatMat( const double &alpha, const ParallelCsrMatrixClass<double> &A, char transa, const ParallelCsrMatrixClass<double> &B, char transb, const double &beta);
   template int ParallelCsrMatrixClass<complexs>::MatMat( const complexs &alpha, const ParallelCsrMatrixClass<complexs> &A, char transa, const ParallelCsrMatrixClass<complexs> &B, char transb, const complexs &beta);
   template int ParallelCsrMatrixClass<complexd>::MatMat( const complexd &alpha, const ParallelCsrMatrixClass<complexd> &A, char transa, const ParallelCsrMatrixClass<complexd> &B, char transb, const complexd &beta);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<T> &parcsrmat_out)
   {

#ifdef PARGEMSLR_CUDA
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Extracting parallel csr sub-matrix only availiable on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      if(rows.GetDataLocation() == kMemoryDevice || cols.GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Working vectors for extracting parallel csr sub-matrix can be on the host memory.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
#endif
      int                        i, j, j1, j2, k;
      int                        nrows_local, ncols_local, A_noffd, noffd_new, row, col;
      long int                   B_row_start, B_col_start;
      vector_long                row_start, col_start, nCdisps, nRdisps, marker_offd;
      vector_int                 rows_local, cols_local, marker_col, marker_row;
      std::vector<vector_long>   send_buff_vec2, recv_buff_vec2;
      
      nrows_local                            = rows.GetLengthLocal();
      ncols_local                            = cols.GetLengthLocal();
      A_noffd                                = this->_offd_map_v.GetLengthLocal();
      
      /* create new parallel csr matrix */
      parcsrmat_out.Setup(nrows_local, ncols_local, *this);
      
      /* we call this matrix as A */
      int *A_offd_i                          = this->_offd_mat.GetI();
      int *A_offd_j                          = this->_offd_mat.GetJ();
      T *A_offd_data                         = this->_offd_mat.GetData();
      
      /* we call the output as B */
      CsrMatrixClass<T> &B_diag              = parcsrmat_out.GetDiagMat();
      CsrMatrixClass<T> &B_offd              = parcsrmat_out.GetOffdMat();
      
      vector_long &B_offd_map                = parcsrmat_out.GetOffdMap();
      
      int           np, myid;
      MPI_Comm      comm;
      
      this->GetMpiInfo(np, myid, comm);
      
      /* setup matvec helper to use the information */
      this->SetupMatvec();
      
      /* first convert global permutation into local permutation */
      rows_local.Setup(nrows_local);
      cols_local.Setup(ncols_local);
      
      for(i = 0 ; i < nrows_local ; i ++)
      {
         rows_local[i] = (int)(rows[i]-this->_nrow_start);
      }
      
      for(i = 0 ; i < ncols_local ; i ++)
      {
         cols_local[i] = (int)(cols[i]-this->_ncol_start);
      }
      
      /* get the diagonal part first */
      this->_diag_mat.SubMatrix(rows_local, cols_local, kMemoryHost, B_diag);
      
      /* Get global displacement */
      nCdisps.Setup(np+1);
      nRdisps.Setup(np+1);
      
      B_row_start = parcsrmat_out.GetRowStartGlobal();
      B_col_start = parcsrmat_out.GetColStartGlobal();
      
      PARGEMSLR_MPI_CALL( PargemslrMpiAllgather( &(B_row_start), 1, nRdisps.GetData(), comm) );
      PARGEMSLR_MPI_CALL( PargemslrMpiAllgather( &(B_col_start), 1, nCdisps.GetData(), comm) );
      
      nCdisps[np] = parcsrmat_out.GetNumRowsGlobal();
      nRdisps[np] = parcsrmat_out.GetNumColsGlobal();
      
      send_buff_vec2.resize(this->_comm_helper._send_to_v.GetLengthLocal());
      recv_buff_vec2.resize(this->_comm_helper._recv_from_v.GetLengthLocal());
      
      marker_row.Setup(this->_nrow_local);
      marker_row.Fill(-1);
      marker_col.Setup(this->_ncol_local);
      marker_col.Fill(-1);
      marker_offd.Setup(A_noffd);
      marker_offd.Fill(-2);
      
      for(i = 0 ; i < nrows_local ; i ++)
      {
         marker_row[rows_local[i]] = i;
      }
      for(i = 0 ; i < ncols_local ; i ++)
      {
         marker_col[cols_local[i]] = i;
      }
      
      /* mark send those used columns */
      j1 = (int)this->_comm_helper._send_idx_v2.size();
      for(i = 0 ; i < j1 ; i ++)
      {
         j2 = this->_comm_helper._send_idx_v2[i].GetLengthLocal();
         send_buff_vec2[i].Setup(j2);
         
         for(j = 0 ; j < j2 ; j ++)
         {
            col = this->_comm_helper._send_idx_v2[i][j];
            if(marker_col[col] >= 0)
            {
               /* we keep this column */
               send_buff_vec2[i][j] = marker_col[col] + nCdisps[myid];
            }
            else
            {
               send_buff_vec2[i][j] = -1;
            }
         }
      }
      
      j1 = (int)this->_comm_helper._recv_idx_v2.size();
      for(i = 0 ; i < j1 ; i ++)
      {
         j2 = this->_comm_helper._recv_idx_v2[i].GetLengthLocal();
         recv_buff_vec2[i].Setup(j2);
      }
      
      /* start communication */
      this->_comm_helper.template DataTransferStart<long int, vector_long>(send_buff_vec2, kMemoryHost);
      
      for(i = 0 ; i < nrows_local ; i ++)
      {
         row = rows_local[i];
         //active only some of the columns
         j1 = A_offd_i[row];
         j2 = A_offd_i[row+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = A_offd_j[j];
            marker_offd[col] = -1;
         }
      }
      
      /* finish communication */
      this->_comm_helper.template DataTransferOver<long int, vector_long>(recv_buff_vec2, kMemoryHost);
      
      j1 = (int)this->_comm_helper._recv_idx_v2.size();
      noffd_new = 0;
      for(i = 0 ; i < j1 ; i ++)
      {
         j2 = this->_comm_helper._recv_idx_v2[i].GetLengthLocal();
         for(j = 0 ; j < j2 ; j ++)
         {
            col = recv_buff_vec2[i][j];
            if(col >= 0)
            {
               k = this->_comm_helper._recv_idx_v2[i][j];
               if(marker_offd[k] > -2)
               {
                  // this is an active column
                  marker_offd[k] = col;
                  noffd_new++;
               }
            }
         }
      }
      
      B_offd_map.Setup(noffd_new);
      
      /* now build the offd */
      B_offd.Setup( nrows_local, noffd_new, this->_offd_mat.GetNumNonzeros());
      
      /* get pointer to B_offd */
      int *B_offd_i                          = B_offd.GetI();
      int *B_offd_j                          = B_offd.GetJ();
      T *B_offd_data                         = B_offd.GetData();
      
      j = 0;
      for( i = 0 ; i < A_noffd ; i ++)
      {
         if(marker_offd[i] >= 0)
         {
            B_offd_map[j] = marker_offd[i];
            marker_offd[i] = j++;
         }
      }
      
      B_offd_i[0] = 0;
      for( i = 0 ; i < nrows_local ; i ++)
      {
         B_offd_i[i+1] = B_offd_i[i];
         row = rows_local[i];
         j1 = A_offd_i[row];
         j2 = A_offd_i[row+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = A_offd_j[j];
            if(marker_offd[col] >= 0)
            {
               B_offd_j[B_offd_i[i+1]] = (int) marker_offd[col];
               B_offd_data[B_offd_i[i+1]++] = A_offd_data[j];
            }
         }
      }
      B_offd.SetNumNonzeros();
      
      nCdisps.Clear();
      nRdisps.Clear();
      marker_offd.Clear();
      marker_col.Clear();
      marker_row.Clear();
      
      j1 = (int) send_buff_vec2.size();
      for(i = 0 ; i < j1 ; i ++)
      {
         send_buff_vec2[i].Clear();
      }
      std::vector<vector_long>().swap(send_buff_vec2);
      
      j1 = (int) recv_buff_vec2.size();
      for(i = 0 ; i < j1 ; i ++)
      {
         recv_buff_vec2[i].Clear();
      }
      std::vector<vector_long>().swap(recv_buff_vec2);
      
      parcsrmat_out.MoveData(location);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<float> &parcsrmat_out);
   template int ParallelCsrMatrixClass<double>::SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<double> &parcsrmat_out);
   template int ParallelCsrMatrixClass<complexs>::SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<complexs> &parcsrmat_out);
   template int ParallelCsrMatrixClass<complexd>::SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<complexd> &parcsrmat_out);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Transpose()
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Transpose only works on host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      ParallelCsrMatrixClass<T> AT = std::move(*this);
      
      ParallelCsrMatrixTransposeHost( AT, *this);
      
      AT.Clear();
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Transpose();
   template int ParallelCsrMatrixClass<double>::Transpose();
   template int ParallelCsrMatrixClass<complexs>::Transpose();
   template int ParallelCsrMatrixClass<complexd>::Transpose();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Transpose(ParallelCsrMatrixClass<T> &AT)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Transpose only works on host.");
         return PARGEMSLR_ERROR_MEMORY_LOCATION;
      }
      
      ParallelCsrMatrixTransposeHost( *this, AT);
      
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::Transpose(ParallelCsrMatrixClass<float> &AT);
   template int ParallelCsrMatrixClass<double>::Transpose(ParallelCsrMatrixClass<double> &AT);
   template int ParallelCsrMatrixClass<complexs>::Transpose(ParallelCsrMatrixClass<complexs> &AT);
   template int ParallelCsrMatrixClass<complexd>::Transpose(ParallelCsrMatrixClass<complexd> &AT);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, T alphax, T alphay, T alphaz, T shift, parallel_log &parlog, bool rand_perturb)
   {
      this->Clear();
      
      MPI_Comm comm;
      int np, myid;
      
      parlog.GetMpiInfo( np, myid, comm);
      
      PARGEMSLR_CHKERR( np != dx * dy * dz );
      PARGEMSLR_CHKERR(nx < dx);
      PARGEMSLR_CHKERR(ny < dy);
      PARGEMSLR_CHKERR(nz < dz);
      
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      
      bool                 two_D;
      long int             idx;
      int                  i, j, k;
      int                  ldx, ldy, ldz;
      int                  ii, offdsize, lidx;
      T                    vd, vxm, vxp, vym, vyp, vzm, vzp;
      RealDataType         ptrb, shift_abs; 
      CooMatrixClass<T>    Acoo_diag, Acoo_offd;
      T                    mone = -1.0;
      
      
      /* global displacement */
      int                  dxy;
      int                  ldx2, ldy2, ldz2;
      int                  numx_base, numy_base, numz_base;
      int                  numx_extra, numy_extra, numz_extra;
      vector_int           numxs, numys, numzs;
      vector_int           dispxs, dispys, dispzs;
      vector_long          dispglobal;
      
      
      dxy = dx * dy;
      
      numx_base  = nx / dx;
      numy_base  = ny / dy;
      numz_base  = nz / dz;
      numx_extra  = nx % dx;
      numy_extra  = ny % dy;
      numz_extra  = nz % dz;
      
      numxs.Setup(dx);
      dispxs.Setup(dx+1);
      numys.Setup(dy);
      dispys.Setup(dy+1);
      numzs.Setup(dz);
      dispzs.Setup(dz+1);
      
      dispglobal.Setup(np+1);
      
      dispxs[0] = 0;
      for(i = 0 ; i < numx_extra ; i ++)
      {
         numxs[i] = numx_base+1;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      for(i = numx_extra ; i < dx ; i ++)
      {
         numxs[i] = numx_base;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      
      dispys[0] = 0;
      for(i = 0 ; i < numy_extra ; i ++)
      {
         numys[i] = numy_base+1;
         dispys[i+1] = numys[i] + dispys[i];
      }
      for(i = numy_extra ; i < dy ; i ++)
      {
         numys[i] = numy_base;
         dispys[i+1] = numys[i] + dispys[i];
      }
      
      dispzs[0] = 0;
      for(i = 0 ; i < numz_extra ; i ++)
      {
         numzs[i] = numz_base+1;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      for(i = numz_extra ; i < dz ; i ++)
      {
         numzs[i] = numz_base;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      
      /* We assign the nodes to procs follow the following order:
       * x first, y next, z last.
       * Thus, a location (x, y, z) corresponds to myid = x + y*dx + z*(dx*dy)
       * If we have myid, we can compute x, y, and z follow:
       * x = myid % dx;
       * z = myid/(dx*dy) -> x + y*dx = myid % (dx*dy)
       */

      /* first update the global disp array */
      dispglobal[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         ldx2   = i % dx;
         ldz2   = i / dxy;
         ldy2   = (i - ldz2 * dxy) / dx;
         dispglobal[i+1] = dispglobal[i] + (long int)numxs[ldx2]*numys[ldy2]*numzs[ldz2];
      }
      
      ldx   = myid % dx;
      ldz   = myid / dxy;
      ldy   = (myid - ldz * dxy) / dx;
      
      /* 2D or 3D */
      two_D       = (PargemslrMin(PargemslrMin(nx, ny), nz) == 1) ? true : false;
      
      /* create values */
      vd          = two_D ? 4.0 : 6.0;
      vd         -= shift;
      vxm         = mone + alphax;
      vxp         = mone - alphax;
      vym         = mone + alphay;
      vyp         = mone - alphay;
      vzm         = mone + alphaz;
      vzp         = mone - alphaz;
      
      shift_abs = PargemslrAbs(shift);
      
      /* next step is to get the problem size 
       * note that now we might have uneven size
       */
      this->_nrow_global      = (long int)nx * ny * nz;
      this->_nrow_local       = numxs[ldx]*numys[ldy]*numzs[ldz];
      this->_nrow_start       = dispglobal[myid];
      this->_ncol_global      = this->_nrow_global;
      this->_ncol_local       = this->_nrow_local;
      this->_ncol_start       = this->_nrow_start;
      
      /* set the default partition, init every node to be interior, update later */
      this->_separator_ndom   = np;
      this->_separator_domi.Setup(this->_nrow_local);
      this->_separator_domi.Fill(myid);
      
      /* copy comm */
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup( comm, this->_comm));
      this->_size = np;
      this->_rank = myid;
      
      /*********************************
       * step 1: insert diagonal entries
       *********************************/
       
      /* create diagonal blocks */
      if(two_D)
      {
         // 5-pt
         Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 5*this->_nrow_local);
      }
      else
      {
         // 7-pt
         Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 7*this->_nrow_local);
      }
      
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               if(rand_perturb)
               {
                  PargemslrValueRandHost(ptrb);
                  Acoo_diag.PushBack(ii, ii, vd - shift_abs*ptrb);
               }
               else
               {
                  Acoo_diag.PushBack(ii, ii, vd);
               }
            }
         }
      } 
      
      /********************************************
       * step 2: insert connection for 6 directions
       ********************************************/
      
      /* x- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 1 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-1, vxm);
            }
         }
      }
      /* x+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx]-1 ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+1, vxp);
            }
         }
      } 
      
      /* y- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 1 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx], vym);
            }
         }
      } 
      /* y+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] - 1 ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx], vyp);
            }
         }
      } 
      
      /* z- side */
      for(i = 1 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx] * numys[ldy], vzm);
            }
         }
      } 
      /* z+ side */
      for(i = 0 ; i < numzs[ldz]-1 ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx] * numys[ldy], vzp);
            }
         }
      } 
      
      /*********************************
       * step 3: build offdiagonal matrix
       *********************************/ 
      
      offdsize = 0;
      if(ldx > 0)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-1);
         this->_comm_helper._send_to_v.PushBack(myid-1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-1);
         this->_trans_comm_helper._send_to_v.PushBack(myid-1);
      }
      if(ldx < dx - 1)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+1);
         this->_comm_helper._send_to_v.PushBack(myid+1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+1);
         this->_trans_comm_helper._send_to_v.PushBack(myid+1);
      }
      if(ldy > 0)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_comm_helper._send_to_v.PushBack(myid-dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx);
      }
      if(ldy < dy - 1)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_comm_helper._send_to_v.PushBack(myid+dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx);
      }
      if(ldz > 0)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx*dy);
      }
      if(ldz < dz - 1)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx*dy);
      }
      
      this->_offd_map_v.Setup(offdsize);
      Acoo_offd.Setup(this->_nrow_local, offdsize, offdsize);
      
      lidx = 0;
      if(ldz > 0)
      {
         /* connection with the lower block */
         idx = dispglobal[myid - (dx*dy)] + numxs[ldx] * numys[ldy] * (numzs[ldz-1] - 1); 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numys[ldy]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
         
      }
      if(ldy > 0)
      {
         /* connection with the front block */
         idx = dispglobal[myid - dx] + numxs[ldx] * (numys[ldy-1] - 1); 
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vym);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy-1] - 1);
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx > 0)
      {
         /* connection with the left block */
         idx = dispglobal[myid - 1] + numxs[ldx-1] - 1;
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               Acoo_offd.PushBack(ii, lidx, vxm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx-1];
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx < dx - 1)
      {
         /* connection with the right block */
         idx = dispglobal[myid + 1];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               Acoo_offd.PushBack(ii, lidx, vxp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx+1];
            }
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldy < dy - 1)
      {
         /* connection with the back block */
         idx = dispglobal[myid + dx];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vyp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy+1] - 1);
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldz < dz - 1)
      {
         /* connection with the upper block */
         idx = dispglobal[myid + (dx*dy)]; 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         
         /* mark the second half */
         for(i = numys[ldy]/2 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      Acoo_diag.ToCsr( kMemoryHost, this->_diag_mat);
      Acoo_offd.ToCsr( kMemoryHost, this->_offd_mat);
      
      this->_is_offd_map_sorted = true;
      this->_comm_helper._is_nbhd_built = true;
      this->_trans_comm_helper._is_nbhd_built = true;
      
      Acoo_diag.Clear();
      Acoo_offd.Clear();
      
      numxs.Clear();
      dispxs.Clear();
      numys.Clear();
      dispys.Clear();
      numzs.Clear();
      dispzs.Clear();
      dispglobal.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, float alphax, float alphay, float alphaz, float shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<double>::Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, double alphax, double alphay, double alphaz, double shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<complexs>::Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, complexs alphax, complexs alphay, complexs alphaz, complexs shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<complexd>::Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, complexd alphax, complexd alphay, complexd alphaz, complexd shift, parallel_log &parlog, bool rand_perturb);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, T alphax, T alphay, T alphaz, T shift, parallel_log &parlog, bool rand_perturb)
   {
      this->Clear();
      
      MPI_Comm comm;
      int np, myid;
      
      parlog.GetMpiInfo( np, myid, comm);
      
      /* switch to default option if parameters incorrect */
      if( (np != dx * dy * dz) || ((d2x % dx) != 0) || ((d2y % dy) != 0) || ((d2z % dz) != 0)
            || (nx < d2x) || (ny < d2y) || (nz < d2z) )
      {
         if(myid == 0)
         {
            PARGEMSLR_WARNING("Invalid Laplacian parameters. Switch to default option.");
         }
         return this->Laplacian(nx, ny, nz, 1, np, 1, alphax, alphay, alphaz, shift, parlog, rand_perturb);
      }
      
      typedef typename std::conditional<PargemslrIsDoublePrecision<T>::value, double, float>::type RealDataType;
      
      bool                 two_D;
      long int             idx;
      int                  i, j, k;
      int                  ldx, ldy, ldz;
      int                  ii, offdsize, lidx;
      T                    vd, vxm, vxp, vym, vyp, vzm, vzp;
      RealDataType         ptrb, shift_abs;
      CooMatrixClass<T>    Acoo_diag, Acoo_offd;
      T                    mone = -1.0;
      
      
      /* global displacement */
      int                  dxy;
      int                  ldx2, ldy2, ldz2;
      int                  numx_base, numy_base, numz_base;
      int                  numx_extra, numy_extra, numz_extra;
      vector_int           numxs, numys, numzs;
      vector_int           dispxs, dispys, dispzs;
      vector_long          dispglobal;
      
      
      dxy = dx * dy;
      
      numx_base  = nx / dx;
      numy_base  = ny / dy;
      numz_base  = nz / dz;
      numx_extra  = nx % dx;
      numy_extra  = ny % dy;
      numz_extra  = nz % dz;
      
      numxs.Setup(dx);
      dispxs.Setup(dx+1);
      numys.Setup(dy);
      dispys.Setup(dy+1);
      numzs.Setup(dz);
      dispzs.Setup(dz+1);
      
      dispglobal.Setup(np+1);
      
      dispxs[0] = 0;
      for(i = 0 ; i < numx_extra ; i ++)
      {
         numxs[i] = numx_base+1;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      for(i = numx_extra ; i < dx ; i ++)
      {
         numxs[i] = numx_base;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      
      dispys[0] = 0;
      for(i = 0 ; i < numy_extra ; i ++)
      {
         numys[i] = numy_base+1;
         dispys[i+1] = numys[i] + dispys[i];
      }
      for(i = numy_extra ; i < dy ; i ++)
      {
         numys[i] = numy_base;
         dispys[i+1] = numys[i] + dispys[i];
      }
      
      dispzs[0] = 0;
      for(i = 0 ; i < numz_extra ; i ++)
      {
         numzs[i] = numz_base+1;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      for(i = numz_extra ; i < dz ; i ++)
      {
         numzs[i] = numz_base;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      
      /* We assign the nodes to procs follow the following order:
       * x first, y next, z last.
       * Thus, a location (x, y, z) corresponds to myid = x + y*dx + z*(dx*dy)
       * If we have myid, we can compute x, y, and z follow:
       * x = myid % dx;
       * z = myid/(dx*dy) -> x + y*dx = myid % (dx*dy)
       */

      /* first update the global disp array */
      dispglobal[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         ldx2   = i % dx;
         ldz2   = i / dxy;
         ldy2   = (i - ldz2 * dxy) / dx;
         dispglobal[i+1] = dispglobal[i] + (long int)numxs[ldx2]*numys[ldy2]*numzs[ldz2];
      }
      
      ldx   = myid % dx;
      ldz   = myid / dxy;
      ldy   = (myid - ldz * dxy) / dx;
      
      /* 2D or 3D */
      two_D       = (PargemslrMin(PargemslrMin(nx, ny), nz) == 1) ? true : false;
      
      /* create values */
      vd          = two_D ? 4.0 : 6.0;
      vd         -= shift;
      vxm         = mone + alphax;
      vxp         = mone - alphax;
      vym         = mone + alphay;
      vyp         = mone - alphay;
      vzm         = mone + alphaz;
      vzp         = mone - alphaz;
      
      shift_abs = PargemslrAbs(shift);
      
      /* next step is to get the problem size 
       * note that now we might have uneven size
       */
      this->_nrow_global      = (long int)nx * ny * nz;
      this->_nrow_local       = numxs[ldx]*numys[ldy]*numzs[ldz];
      this->_nrow_start       = dispglobal[myid];
      this->_ncol_global      = this->_nrow_global;
      this->_ncol_local       = this->_nrow_local;
      this->_ncol_start       = this->_nrow_start;
      
      /* copy comm */
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup( comm, this->_comm));
      this->_size = np;
      this->_rank = myid;
      
      /*********************************
       * step 1: insert diagonal entries
       *********************************/
       
      /* create diagonal blocks */
      if(two_D)
      {
         // 5-pt
         Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 5*this->_nrow_local);
      }
      else
      {
         // 7-pt
         Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 7*this->_nrow_local);
      }
      
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               if(rand_perturb)
               {
                  PargemslrValueRandHost(ptrb);
                  Acoo_diag.PushBack(ii, ii, vd - shift_abs*ptrb);
               }
               else
               {
                  Acoo_diag.PushBack(ii, ii, vd);
               }
            }
         }
      } 
      
      /********************************************
       * step 2: insert connection for 6 directions
       ********************************************/
      
      /* x- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 1 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-1, vxm);
            }
         }
      }
      /* x+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx]-1 ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+1, vxp);
            }
         }
      } 
      
      /* y- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 1 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx], vym);
            }
         }
      } 
      /* y+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] - 1 ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx], vyp);
            }
         }
      } 
      
      /* z- side */
      for(i = 1 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx] * numys[ldy], vzm);
            }
         }
      } 
      /* z+ side */
      for(i = 0 ; i < numzs[ldz]-1 ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx] * numys[ldy], vzp);
            }
         }
      } 
      
      /*********************************
       * step 3: build offdiagonal matrix
       *********************************/ 
      
      offdsize = 0;
      if(ldx > 0)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-1);
         this->_comm_helper._send_to_v.PushBack(myid-1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-1);
         this->_trans_comm_helper._send_to_v.PushBack(myid-1);
      }
      if(ldx < dx - 1)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+1);
         this->_comm_helper._send_to_v.PushBack(myid+1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+1);
         this->_trans_comm_helper._send_to_v.PushBack(myid+1);
      }
      if(ldy > 0)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_comm_helper._send_to_v.PushBack(myid-dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx);
      }
      if(ldy < dy - 1)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_comm_helper._send_to_v.PushBack(myid+dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx);
      }
      if(ldz > 0)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx*dy);
      }
      if(ldz < dz - 1)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx*dy);
      }
      
      this->_offd_map_v.Setup(offdsize);
      Acoo_offd.Setup(this->_nrow_local, offdsize, offdsize);
      
      lidx = 0;
      if(ldz > 0)
      {
         /* connection with the lower block */
         idx = dispglobal[myid - (dx*dy)] + numxs[ldx] * numys[ldy] * (numzs[ldz-1] - 1); 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
      }
      if(ldy > 0)
      {
         /* connection with the front block */
         idx = dispglobal[myid - dx] + numxs[ldx] * (numys[ldy-1] - 1); 
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vym);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy-1] - 1);
         }
      }
      if(ldx > 0)
      {
         /* connection with the left block */
         idx = dispglobal[myid - 1] + numxs[ldx-1] - 1;
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               Acoo_offd.PushBack(ii, lidx, vxm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx-1];
            }
         }
      }
      if(ldx < dx - 1)
      {
         /* connection with the right block */
         idx = dispglobal[myid + 1];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               Acoo_offd.PushBack(ii, lidx, vxp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx+1];
            }
         }
      }
      
      if(ldy < dy - 1)
      {
         /* connection with the back block */
         idx = dispglobal[myid + dx];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vyp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy+1] - 1);
         }
      }
      
      if(ldz < dz - 1)
      {
         /* connection with the upper block */
         idx = dispglobal[myid + (dx*dy)]; 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
      }
      
      Acoo_diag.ToCsr( kMemoryHost, this->_diag_mat);
      Acoo_offd.ToCsr( kMemoryHost, this->_offd_mat);
      
      this->_is_offd_map_sorted = true;
      this->_comm_helper._is_nbhd_built = true;
      this->_trans_comm_helper._is_nbhd_built = true;
      
      Acoo_diag.Clear();
      Acoo_offd.Clear();
      
      /* prepare the separator */
      int         i1, j1, k1;
      int         ndomxy, ndomx, ndomy, ndomz, ndom, color;
      vector_int  ndomxs, ndomys, ndomzs;
      vector_int  ndomxdisps, ndomydisps, ndomzdisps;
      vector_int  globalndomdisp;
      int         nodesdomx_base, nodesdomy_base, nodesdomz_base;
      int         nodesdomx_extra, nodesdomy_extra, nodesdomz_extra;
      
      ndomx = d2x/dx;
      ndomy = d2y/dy;
      ndomz = d2z/dz;
      
      ndomxy = ndomx*ndomy;
      
      ndom = ndomx*ndomy*ndomz;
      
      nodesdomx_base  = numxs[ldx] / ndomx;
      nodesdomy_base  = numys[ldy] / ndomy;
      nodesdomz_base  = numzs[ldz] / ndomz;
      nodesdomx_extra  = numxs[ldx] % ndomx;
      nodesdomy_extra  = numys[ldy] % ndomy;
      nodesdomz_extra  = numzs[ldz] % ndomz;
      
      ndomxs.Setup(ndomx);
      ndomxdisps.Setup(ndomx+1);
      ndomys.Setup(ndomy);
      ndomydisps.Setup(ndomy+1);
      ndomzs.Setup(ndomz);
      ndomzdisps.Setup(ndomz+1);
      
      globalndomdisp.Setup(ndom);
      
      ndomxdisps[0] = 0;
      for(i = 0 ; i < nodesdomx_extra ; i ++)
      {
         ndomxs[i] = nodesdomx_base+1;
         ndomxdisps[i+1] = ndomxs[i] + ndomxdisps[i];
      }
      for(i = nodesdomx_extra ; i < ndomx ; i ++)
      {
         ndomxs[i] = nodesdomx_base;
         ndomxdisps[i+1] = ndomxs[i] + ndomxdisps[i];
      }
      
      ndomydisps[0] = 0;
      for(i = 0 ; i < nodesdomy_extra ; i ++)
      {
         ndomys[i] = nodesdomy_base+1;
         ndomydisps[i+1] = ndomys[i] + ndomydisps[i];
      }
      for(i = nodesdomy_extra ; i < ndomy ; i ++)
      {
         ndomys[i] = nodesdomy_base;
         ndomydisps[i+1] = ndomys[i] + ndomydisps[i];
      }
      
      ndomzdisps[0] = 0;
      for(i = 0 ; i < nodesdomz_extra ; i ++)
      {
         ndomzs[i] = nodesdomz_base+1;
         ndomzdisps[i+1] = ndomzs[i] + ndomzdisps[i];
      }
      for(i = nodesdomz_extra ; i < ndomz ; i ++)
      {
         ndomzs[i] = nodesdomz_base;
         ndomzdisps[i+1] = ndomzs[i] + ndomzdisps[i];
      }
      
      for(i = 0 ; i < ndom ; i ++)
      {
         ldx2   = i % ndomx;
         ldz2   = i / ndomxy;
         ldy2   = (i - ldz2 * ndomxy) / ndomx;
         globalndomdisp[i] = ndomzdisps[ldz2] * numxs[ldx] * numys[ldy] + ndomydisps[ldy2] * numxs[ldx] + ndomxdisps[ldx2];
      }
      
      /* set the default partition, init every node to be interior, update later */
      this->_separator_ndom   = np * ndom;
      this->_separator_domi.Setup(this->_nrow_local);
      
      this->_separator_domi.Fill(-2);
      
      color = 0;
      for(i1 = 0 ; i1 < ndomz ; i1 ++)
      {
         for(j1 = 0 ; j1 < ndomy ; j1 ++)
         {
            for(k1 = 0 ; k1 < ndomx ; k1 ++)
            {
               for(i = 0 ; i < ndomzs[i1] ; i ++)
               {
                  for(j = 0 ; j < ndomys[j1] ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        // ii is the local index
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = color+myid*ndom;
                     }
                  }
               }
               
               /* update boundary */
               if(k1 > 0 || ldx > 0)
               {
                  /* there is one on the left, mark upper half */
                  for(i = 0 ; i < ndomzs[i1]/2 ; i ++)
                  {
                     for(j = 0 ; j < ndomys[j1] ; j ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx];
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(k1 < ndomx - 1 || ldx < dx - 1)
               {
                  /* one on the right, mark lower half */
                  for(i = ndomzs[i1]/2 ; i < ndomzs[i1] ; i ++)
                  {
                     for(j = 0 ; j < ndomys[j1] ; j ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + ndomxs[k1] - 1;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(j1 > 0 || ldy > 0)
               {
                  /* one on the front, mark upper */
                  for(i = 0 ; i < ndomzs[i1]/2 ; i ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
                  
               }
               if(j1 < ndomy - 1 || ldy < dy - 1)
               {
                  /* one on the back, mark lower */
                  for(i = ndomzs[i1]/2 ; i < ndomzs[i1] ; i ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + (ndomys[j1]-1) * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(i1 > 0 || ldz > 0)
               {
                  /* one down, mark upper */
                  for(j = 0 ; j < ndomys[j1]/2 ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(i1 < ndomz - 1 || ldz < dz - 1)
               {
                  /* one up, mark lower */
                  for(j = ndomys[j1]/2 ; j < ndomys[j1] ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + (ndomzs[i1] - 1) * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               color++;
            }
         }
      } 
      
      numxs.Clear();
      dispxs.Clear();
      numys.Clear();
      dispys.Clear();
      numzs.Clear();
      dispzs.Clear();
      dispglobal.Clear();
      
      ndomxs.Clear();
      ndomxdisps.Clear();
      ndomys.Clear();
      ndomydisps.Clear();
      ndomzs.Clear();
      ndomzdisps.Clear();
      
      globalndomdisp.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, float alphax, float alphay, float alphaz, float shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<double>::LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, double alphax, double alphay, double alphaz, double shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<complexs>::LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, complexs alphax, complexs alphay, complexs alphaz, complexs shift, parallel_log &parlog, bool rand_perturb);
   template int ParallelCsrMatrixClass<complexd>::LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, complexd alphax, complexd alphay, complexd alphaz, complexd shift, parallel_log &parlog, bool rand_perturb);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::Helmholtz(int n, int dx, int dy, int dz, T w, parallel_log &parlog)
   {
      this->Clear();
      
      MPI_Comm comm;
      int np, myid;
      
      parlog.GetMpiInfo( np, myid, comm);
      
      PARGEMSLR_CHKERR( np != dx * dy * dz );
      PARGEMSLR_CHKERR(n < dx);
      PARGEMSLR_CHKERR(n < dy);
      PARGEMSLR_CHKERR(n < dz);
      
      long int             idx;
      int                  i, j, k;
      int                  ldx, ldy, ldz;
      int                  ii, offdsize, lidx;
      T                    v, h, c, vd, vxm, vxp, vym, vyp, vzm, vzp; 
      CooMatrixClass<T>    Acoo_diag, Acoo_offd;
      T                    mone = -1.0;
      
      if(n < 1)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      /* this is the grid size */
      h = T(1.0)/T(n-1);
      vd = T(6.0) - w*w*h*h;
      c = T(0.0,2.0)*h*w;
      
      /* global displacement */
      int                  nx, ny, nz;
      int                  dxy;
      int                  ldx2, ldy2, ldz2;
      int                  numx_base, numy_base, numz_base;
      int                  numx_extra, numy_extra, numz_extra;
      vector_int           numxs, numys, numzs;
      vector_int           dispxs, dispys, dispzs;
      vector_long          dispglobal;
      
      nx = n;
      ny = n;
      nz = n;
      
      dxy = dx * dy;
      
      numx_base  = nx / dx;
      numy_base  = ny / dy;
      numz_base  = nz / dz;
      numx_extra  = nx % dx;
      numy_extra  = ny % dy;
      numz_extra  = nz % dz;
      
      numxs.Setup(dx);
      dispxs.Setup(dx+1);
      numys.Setup(dy);
      dispys.Setup(dy+1);
      numzs.Setup(dz);
      dispzs.Setup(dz+1);
      
      dispglobal.Setup(np+1);
      
      dispxs[0] = 0;
      for(i = 0 ; i < numx_extra ; i ++)
      {
         numxs[i] = numx_base+1;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      for(i = numx_extra ; i < dx ; i ++)
      {
         numxs[i] = numx_base;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      
      dispys[0] = 0;
      for(i = 0 ; i < numy_extra ; i ++)
      {
         numys[i] = numy_base+1;
         dispys[i+1] = numys[i] + dispys[i];
      }
      for(i = numy_extra ; i < dy ; i ++)
      {
         numys[i] = numy_base;
         dispys[i+1] = numys[i] + dispys[i];
      }
      
      dispzs[0] = 0;
      for(i = 0 ; i < numz_extra ; i ++)
      {
         numzs[i] = numz_base+1;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      for(i = numz_extra ; i < dz ; i ++)
      {
         numzs[i] = numz_base;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      
      /* We assign the nodes to procs follow the following order:
       * x first, y next, z last.
       * Thus, a location (x, y, z) corresponds to myid = x + y*dx + z*(dx*dy)
       * If we have myid, we can compute x, y, and z follow:
       * x = myid % dx;
       * z = myid/(dx*dy) -> x + y*dx = myid % (dx*dy)
       */

      /* first update the global disp array */
      dispglobal[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         ldx2   = i % dx;
         ldz2   = i / dxy;
         ldy2   = (i - ldz2 * dxy) / dx;
         dispglobal[i+1] = dispglobal[i] + (long int)numxs[ldx2]*numys[ldy2]*numzs[ldz2];
      }
      
      ldx   = myid % dx;
      ldz   = myid / dxy;
      ldy   = (myid - ldz * dxy) / dx;
      
      /* next step is to get the problem size 
       * note that now we might have uneven size
       */
      this->_nrow_global      = (long int)nx * ny * nz;
      this->_nrow_local       = numxs[ldx]*numys[ldy]*numzs[ldz];
      this->_nrow_start       = dispglobal[myid];
      this->_ncol_global      = this->_nrow_global;
      this->_ncol_local       = this->_nrow_local;
      this->_ncol_start       = this->_nrow_start;
      
      /* set the default partition, init every node to be interior, update later */
      this->_separator_ndom   = np;
      this->_separator_domi.Setup(this->_nrow_local);
      this->_separator_domi.Fill(myid);
      
      /* copy comm */
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup( comm, this->_comm));
      this->_size = np;
      this->_rank = myid;
      
      /*********************************
       * step 1: insert diagonal entries
       *********************************/
       
      /* create diagonal blocks */
      Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 7*this->_nrow_local);
      
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               v = vd;
               
               if(i == 0 && ldz == 0)
               {
                  /* z == 0 */
                  v = v - c;
               }
               if(i == numzs[ldz]-1 && ldz == dz-1)
               {
                  /* z == 1 */
                  v = v - c;
               }
               if(j == 0 && ldy == 0)
               {
                  /* y == 0 */
                  v = v - c;
               }
               if(j == numys[ldy]-1 && ldy == dy-1)
               {
                  /* y == 1 */
                  v = v - c;
               }
               if(k == 0 && ldx == 0)
               {
                  /* x == 0 */
                  v = v - c;
               }
               if(k == numxs[ldx]-1 && ldx == dx-1)
               {
                  /* x == 1 */
                  v = v - c;
               }
               
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii, v);
            }
         }
      } 
      
      /********************************************
       * step 2: insert connection for 6 directions
       ********************************************/
      
      /* x- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 1 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vxm  = mone;
               if(k == numxs[ldx]-1 && ldx == dx-1)
               {
                  /* x == 1 */
                  vxm = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-1, vxm);
            }
         }
      }
      /* x+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx]-1 ; k ++)
            {
               // ii is the local index
               vxp = mone;
               if(k == 0 && ldx == 0)
               {
                  /* x == 0 */
                  vxp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+1, vxp);
            }
         }
      } 
      
      /* y- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 1 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vym = mone;
               if(j == numys[ldy]-1 && ldy == dy-1)
               {
                  /* y == 1 */
                  vym = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx], vym);
            }
         }
      } 
      /* y+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] - 1 ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vyp = mone;
               if(j == 0 && ldy == 0)
               {
                  /* y == 0 */
                  vyp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx], vyp);
            }
         }
      } 
      
      /* z- side */
      for(i = 1 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vzm = mone;
               if(i == numzs[ldz]-1 && ldz == dz-1)
               {
                  /* z == 1 */
                  vzm = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx] * numys[ldy], vzm);
            }
         }
      } 
      /* z+ side */
      for(i = 0 ; i < numzs[ldz]-1 ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vzp = mone;
               if(i == 0 && ldz == 0)
               {
                  /* z == 0 */
                  vzp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx] * numys[ldy], vzp);
            }
         }
      } 
      
      /*********************************
       * step 3: build offdiagonal matrix
       *********************************/ 
      
      offdsize = 0;
      if(ldx > 0)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-1);
         this->_comm_helper._send_to_v.PushBack(myid-1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-1);
         this->_trans_comm_helper._send_to_v.PushBack(myid-1);
      }
      if(ldx < dx - 1)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+1);
         this->_comm_helper._send_to_v.PushBack(myid+1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+1);
         this->_trans_comm_helper._send_to_v.PushBack(myid+1);
      }
      if(ldy > 0)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_comm_helper._send_to_v.PushBack(myid-dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx);
      }
      if(ldy < dy - 1)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_comm_helper._send_to_v.PushBack(myid+dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx);
      }
      if(ldz > 0)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx*dy);
      }
      if(ldz < dz - 1)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx*dy);
      }
      
      this->_offd_map_v.Setup(offdsize);
      Acoo_offd.Setup(this->_nrow_local, offdsize, offdsize);
      
      /* create values */
      vxp         = mone;
      vym         = mone;
      vyp         = mone;
      vzm         = mone;
      vzp         = mone;
      
      lidx = 0;
      if(ldz > 0)
      {
         /* connection with the lower block */
         idx = dispglobal[myid - (dx*dy)] + numxs[ldx] * numys[ldy] * (numzs[ldz-1] - 1); 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numys[ldy]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
         
      }
      if(ldy > 0)
      {
         /* connection with the front block */
         idx = dispglobal[myid - dx] + numxs[ldx] * (numys[ldy-1] - 1); 
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vym);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy-1] - 1);
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx > 0)
      {
         /* connection with the left block */
         idx = dispglobal[myid - 1] + numxs[ldx-1] - 1;
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               Acoo_offd.PushBack(ii, lidx, vxm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx-1];
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx < dx - 1)
      {
         /* connection with the right block */
         idx = dispglobal[myid + 1];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               Acoo_offd.PushBack(ii, lidx, vxp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx+1];
            }
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldy < dy - 1)
      {
         /* connection with the back block */
         idx = dispglobal[myid + dx];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vyp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy+1] - 1);
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldz < dz - 1)
      {
         /* connection with the upper block */
         idx = dispglobal[myid + (dx*dy)]; 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         
         /* mark the second half */
         for(i = numys[ldy]/2 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      Acoo_diag.ToCsr( kMemoryHost, this->_diag_mat);
      Acoo_offd.ToCsr( kMemoryHost, this->_offd_mat);
      
      this->_is_offd_map_sorted = true;
      this->_comm_helper._is_nbhd_built = true;
      this->_trans_comm_helper._is_nbhd_built = true;
      
      Acoo_diag.Clear();
      Acoo_offd.Clear();
      
      numxs.Clear();
      dispxs.Clear();
      numys.Clear();
      dispys.Clear();
      numzs.Clear();
      dispzs.Clear();
      dispglobal.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<complexs>::Helmholtz(int n, int dx, int dy, int dz, complexs w, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::Helmholtz(int n, int dx, int dy, int dz, complexd w, parallel_log &parlog);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::HelmholtzWithPartition(int n, int dx, int dy, int dz, int d2x, int d2y, int d2z, T w, parallel_log &parlog)
   {
      this->Clear();
      
      MPI_Comm comm;
      int np, myid;
      
      parlog.GetMpiInfo( np, myid, comm);
      
      
      /* switch to default option if parameters incorrect */
      if( (np != dx * dy * dz) || ((d2x % dx) != 0) || ((d2y % dy) != 0) || ((d2z % dz) != 0)
            || (n < d2x) || (n < d2y) || (n < d2z) )
      {
         if(myid == 0)
         {
            PARGEMSLR_WARNING("Invalid Helmholtz parameters. Switch to default option.");
         }
         return this->Helmholtz(n, 1, 1, np, w, parlog );
      }
      
      
      long int             idx;
      int                  i, j, k;
      int                  ldx, ldy, ldz;
      int                  ii, offdsize, lidx;
      T                    v, h, c, vd, vxm, vxp, vym, vyp, vzm, vzp; 
      CooMatrixClass<T>    Acoo_diag, Acoo_offd;
      T                    mone = -1.0;
      
      if(n < 1)
      {
         return PARGEMSLR_SUCCESS;
      }
      
      /* this is the grid size */
      h = T(1.0)/T(n-1);
      vd = T(6.0) - w*w*h*h;
      c = T(0.0,2.0)*h*w;
      
      /* global displacement */
      int                  nx, ny, nz;
      int                  dxy;
      int                  ldx2, ldy2, ldz2;
      int                  numx_base, numy_base, numz_base;
      int                  numx_extra, numy_extra, numz_extra;
      vector_int           numxs, numys, numzs;
      vector_int           dispxs, dispys, dispzs;
      vector_long          dispglobal;
      
      nx = n;
      ny = n;
      nz = n;
      
      dxy = dx * dy;
      
      numx_base  = nx / dx;
      numy_base  = ny / dy;
      numz_base  = nz / dz;
      numx_extra  = nx % dx;
      numy_extra  = ny % dy;
      numz_extra  = nz % dz;
      
      numxs.Setup(dx);
      dispxs.Setup(dx+1);
      numys.Setup(dy);
      dispys.Setup(dy+1);
      numzs.Setup(dz);
      dispzs.Setup(dz+1);
      
      dispglobal.Setup(np+1);
      
      dispxs[0] = 0;
      for(i = 0 ; i < numx_extra ; i ++)
      {
         numxs[i] = numx_base+1;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      for(i = numx_extra ; i < dx ; i ++)
      {
         numxs[i] = numx_base;
         dispxs[i+1] = numxs[i] + dispxs[i];
      }
      
      dispys[0] = 0;
      for(i = 0 ; i < numy_extra ; i ++)
      {
         numys[i] = numy_base+1;
         dispys[i+1] = numys[i] + dispys[i];
      }
      for(i = numy_extra ; i < dy ; i ++)
      {
         numys[i] = numy_base;
         dispys[i+1] = numys[i] + dispys[i];
      }
      
      dispzs[0] = 0;
      for(i = 0 ; i < numz_extra ; i ++)
      {
         numzs[i] = numz_base+1;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      for(i = numz_extra ; i < dz ; i ++)
      {
         numzs[i] = numz_base;
         dispzs[i+1] = numzs[i] + dispzs[i];
      }
      
      /* We assign the nodes to procs follow the following order:
       * x first, y next, z last.
       * Thus, a location (x, y, z) corresponds to myid = x + y*dx + z*(dx*dy)
       * If we have myid, we can compute x, y, and z follow:
       * x = myid % dx;
       * z = myid/(dx*dy) -> x + y*dx = myid % (dx*dy)
       */

      /* first update the global disp array */
      dispglobal[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         ldx2   = i % dx;
         ldz2   = i / dxy;
         ldy2   = (i - ldz2 * dxy) / dx;
         dispglobal[i+1] = dispglobal[i] + (long int)numxs[ldx2]*numys[ldy2]*numzs[ldz2];
      }
      
      ldx   = myid % dx;
      ldz   = myid / dxy;
      ldy   = (myid - ldz * dxy) / dx;
      
      /* next step is to get the problem size 
       * note that now we might have uneven size
       */
      this->_nrow_global      = (long int)nx * ny * nz;
      this->_nrow_local       = numxs[ldx]*numys[ldy]*numzs[ldz];
      this->_nrow_start       = dispglobal[myid];
      this->_ncol_global      = this->_nrow_global;
      this->_ncol_local       = this->_nrow_local;
      this->_ncol_start       = this->_nrow_start;
      
      /* set the default partition, init every node to be interior, update later */
      this->_separator_ndom   = np;
      this->_separator_domi.Setup(this->_nrow_local);
      this->_separator_domi.Fill(myid);
      
      /* copy comm */
      PARGEMSLR_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      PARGEMSLR_MPI_CALL( MPI_Comm_dup( comm, this->_comm));
      this->_size = np;
      this->_rank = myid;
      
      /*********************************
       * step 1: insert diagonal entries
       *********************************/
       
      /* create diagonal blocks */
      Acoo_diag.Setup(this->_nrow_local, this->_nrow_local, 7*this->_nrow_local);
      
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               v = vd;
               
               if(i == 0 && ldz == 0)
               {
                  /* z == 0 */
                  v = v - c;
               }
               if(i == numzs[ldz]-1 && ldz == dz-1)
               {
                  /* z == 1 */
                  v = v - c;
               }
               if(j == 0 && ldy == 0)
               {
                  /* y == 0 */
                  v = v - c;
               }
               if(j == numys[ldy]-1 && ldy == dy-1)
               {
                  /* y == 1 */
                  v = v - c;
               }
               if(k == 0 && ldx == 0)
               {
                  /* x == 0 */
                  v = v - c;
               }
               if(k == numxs[ldx]-1 && ldx == dx-1)
               {
                  /* x == 1 */
                  v = v - c;
               }
               
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii, v);
            }
         }
      } 
      
      /********************************************
       * step 2: insert connection for 6 directions
       ********************************************/
      
      /* x- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 1 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vxm  = mone;
               if(k == numxs[ldx]-1 && ldx == dx-1)
               {
                  /* x == 1 */
                  vxm = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-1, vxm);
            }
         }
      }
      /* x+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx]-1 ; k ++)
            {
               // ii is the local index
               vxp = mone;
               if(k == 0 && ldx == 0)
               {
                  /* x == 0 */
                  vxp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+1, vxp);
            }
         }
      } 
      
      /* y- side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 1 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vym = mone;
               if(j == numys[ldy]-1 && ldy == dy-1)
               {
                  /* y == 1 */
                  vym = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx], vym);
            }
         }
      } 
      /* y+ side */
      for(i = 0 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] - 1 ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vyp = mone;
               if(j == 0 && ldy == 0)
               {
                  /* y == 0 */
                  vyp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx], vyp);
            }
         }
      } 
      
      /* z- side */
      for(i = 1 ; i < numzs[ldz] ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vzm = mone;
               if(i == numzs[ldz]-1 && ldz == dz-1)
               {
                  /* z == 1 */
                  vzm = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii-numxs[ldx] * numys[ldy], vzm);
            }
         }
      } 
      /* z+ side */
      for(i = 0 ; i < numzs[ldz]-1 ; i ++)
      {
         for(j = 0 ; j < numys[ldy] ; j ++)
         {
            for(k = 0 ; k < numxs[ldx] ; k ++)
            {
               // ii is the local index
               vzp = mone;
               if(i == 0 && ldz == 0)
               {
                  /* z == 0 */
                  vzp = T(-2.0);
               }
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
               Acoo_diag.PushBack(ii, ii+numxs[ldx] * numys[ldy], vzp);
            }
         }
      } 
      
      /*********************************
       * step 3: build offdiagonal matrix
       *********************************/ 
      
      offdsize = 0;
      if(ldx > 0)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-1);
         this->_comm_helper._send_to_v.PushBack(myid-1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-1);
         this->_trans_comm_helper._send_to_v.PushBack(myid-1);
      }
      if(ldx < dx - 1)
      {
         offdsize += numys[ldy] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+1);
         this->_comm_helper._send_to_v.PushBack(myid+1);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+1);
         this->_trans_comm_helper._send_to_v.PushBack(myid+1);
      }
      if(ldy > 0)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_comm_helper._send_to_v.PushBack(myid-dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx);
      }
      if(ldy < dy - 1)
      {
         offdsize += numxs[ldx] * numzs[ldz];
         this->_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_comm_helper._send_to_v.PushBack(myid+dx);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx);
      }
      if(ldz > 0)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid-dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid-dx*dy);
      }
      if(ldz < dz - 1)
      {
         offdsize += numxs[ldx] * numys[ldy];
         this->_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_comm_helper._send_to_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._recv_from_v.PushBack(myid+dx*dy);
         this->_trans_comm_helper._send_to_v.PushBack(myid+dx*dy);
      }
      
      this->_offd_map_v.Setup(offdsize);
      Acoo_offd.Setup(this->_nrow_local, offdsize, offdsize);
      
      /* create values */
      vxp         = mone;
      vym         = mone;
      vyp         = mone;
      vzm         = mone;
      vzp         = mone;
      
      lidx = 0;
      if(ldz > 0)
      {
         /* connection with the lower block */
         idx = dispglobal[myid - (dx*dy)] + numxs[ldx] * numys[ldy] * (numzs[ldz-1] - 1); 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numys[ldy]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // add to offd
               ii = 0 * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
         
      }
      if(ldy > 0)
      {
         /* connection with the front block */
         idx = dispglobal[myid - dx] + numxs[ldx] * (numys[ldy-1] - 1); 
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vym);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy-1] - 1);
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + 0 * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx > 0)
      {
         /* connection with the left block */
         idx = dispglobal[myid - 1] + numxs[ldx-1] - 1;
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               Acoo_offd.PushBack(ii, lidx, vxm);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx-1];
            }
         }
         
         /* mark the first half */
         for(i = 0 ; i < numzs[ldz]/2 ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + 0;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      if(ldx < dx - 1)
      {
         /* connection with the right block */
         idx = dispglobal[myid + 1];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               Acoo_offd.PushBack(ii, lidx, vxp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += numxs[ldx+1];
            }
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numys[ldy] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + numxs[ldx] - 1;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldy < dy - 1)
      {
         /* connection with the back block */
         idx = dispglobal[myid + dx];
         
         for(i = 0 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vyp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
            idx += numxs[ldx] * (numys[ldy+1] - 1);
         }
         
         /* mark the second half */
         for(i = numzs[ldz]/2 ; i < numzs[ldz] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = i * numxs[ldx] * numys[ldy] + (numys[ldy] - 1) * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      if(ldz < dz - 1)
      {
         /* connection with the upper block */
         idx = dispglobal[myid + (dx*dy)]; 
         
         for(i = 0 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               Acoo_offd.PushBack(ii, lidx, vzp);
               
               // add the global col number of this row
               this->_offd_map_v[lidx++] = idx;
               idx += 1;
            }
         }
         
         
         /* mark the second half */
         for(i = numys[ldy]/2 ; i < numys[ldy] ; i ++)
         {
            for(j = 0 ; j < numxs[ldx] ; j ++)
            {
               // ii is the local index
               ii = (numzs[ldz] - 1) * numxs[ldx] * numys[ldy] + i * numxs[ldx] + j;
               
               this->_separator_domi[ii] = -1;
            }
         }
      }
      
      Acoo_diag.ToCsr( kMemoryHost, this->_diag_mat);
      Acoo_offd.ToCsr( kMemoryHost, this->_offd_mat);
      
      this->_is_offd_map_sorted = true;
      this->_comm_helper._is_nbhd_built = true;
      this->_trans_comm_helper._is_nbhd_built = true;
      
      Acoo_diag.Clear();
      Acoo_offd.Clear();
      
      
      /* prepare the separator */
      int         i1, j1, k1;
      int         ndomxy, ndomx, ndomy, ndomz, ndom, color;
      vector_int  ndomxs, ndomys, ndomzs;
      vector_int  ndomxdisps, ndomydisps, ndomzdisps;
      vector_int  globalndomdisp;
      int         nodesdomx_base, nodesdomy_base, nodesdomz_base;
      int         nodesdomx_extra, nodesdomy_extra, nodesdomz_extra;
      
      ndomx = d2x/dx;
      ndomy = d2y/dy;
      ndomz = d2z/dz;
      
      ndomxy = ndomx*ndomy;
      
      ndom = ndomx*ndomy*ndomz;
      
      nodesdomx_base  = numxs[ldx] / ndomx;
      nodesdomy_base  = numys[ldy] / ndomy;
      nodesdomz_base  = numzs[ldz] / ndomz;
      nodesdomx_extra  = numxs[ldx] % ndomx;
      nodesdomy_extra  = numys[ldy] % ndomy;
      nodesdomz_extra  = numzs[ldz] % ndomz;
      
      ndomxs.Setup(ndomx);
      ndomxdisps.Setup(ndomx+1);
      ndomys.Setup(ndomy);
      ndomydisps.Setup(ndomy+1);
      ndomzs.Setup(ndomz);
      ndomzdisps.Setup(ndomz+1);
      
      globalndomdisp.Setup(ndom);
      
      ndomxdisps[0] = 0;
      for(i = 0 ; i < nodesdomx_extra ; i ++)
      {
         ndomxs[i] = nodesdomx_base+1;
         ndomxdisps[i+1] = ndomxs[i] + ndomxdisps[i];
      }
      for(i = nodesdomx_extra ; i < ndomx ; i ++)
      {
         ndomxs[i] = nodesdomx_base;
         ndomxdisps[i+1] = ndomxs[i] + ndomxdisps[i];
      }
      
      ndomydisps[0] = 0;
      for(i = 0 ; i < nodesdomy_extra ; i ++)
      {
         ndomys[i] = nodesdomy_base+1;
         ndomydisps[i+1] = ndomys[i] + ndomydisps[i];
      }
      for(i = nodesdomy_extra ; i < ndomy ; i ++)
      {
         ndomys[i] = nodesdomy_base;
         ndomydisps[i+1] = ndomys[i] + ndomydisps[i];
      }
      
      ndomzdisps[0] = 0;
      for(i = 0 ; i < nodesdomz_extra ; i ++)
      {
         ndomzs[i] = nodesdomz_base+1;
         ndomzdisps[i+1] = ndomzs[i] + ndomzdisps[i];
      }
      for(i = nodesdomz_extra ; i < ndomz ; i ++)
      {
         ndomzs[i] = nodesdomz_base;
         ndomzdisps[i+1] = ndomzs[i] + ndomzdisps[i];
      }
      
      for(i = 0 ; i < ndom ; i ++)
      {
         ldx2   = i % ndomx;
         ldz2   = i / ndomxy;
         ldy2   = (i - ldz2 * ndomxy) / ndomx;
         globalndomdisp[i] = ndomzdisps[ldz2] * numxs[ldx] * numys[ldy] + ndomydisps[ldy2] * numxs[ldx] + ndomxdisps[ldx2];
      }
      
      /* set the default partition, init every node to be interior, update later */
      this->_separator_ndom   = np * ndom;
      this->_separator_domi.Setup(this->_nrow_local);
      
      this->_separator_domi.Fill(-2);
      
      color = 0;
      for(i1 = 0 ; i1 < ndomz ; i1 ++)
      {
         for(j1 = 0 ; j1 < ndomy ; j1 ++)
         {
            for(k1 = 0 ; k1 < ndomx ; k1 ++)
            {
               for(i = 0 ; i < ndomzs[i1] ; i ++)
               {
                  for(j = 0 ; j < ndomys[j1] ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        // ii is the local index
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = color+myid*ndom;
                     }
                  }
               }
               
               /* update boundary */
               if(k1 > 0 || ldx > 0)
               {
                  /* there is one on the left, mark upper half */
                  for(i = 0 ; i < ndomzs[i1]/2 ; i ++)
                  {
                     for(j = 0 ; j < ndomys[j1] ; j ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx];
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(k1 < ndomx - 1 || ldx < dx - 1)
               {
                  /* one on the right, mark lower half */
                  for(i = ndomzs[i1]/2 ; i < ndomzs[i1] ; i ++)
                  {
                     for(j = 0 ; j < ndomys[j1] ; j ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + j * numxs[ldx] + ndomxs[k1] - 1;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(j1 > 0 || ldy > 0)
               {
                  /* one on the front, mark upper */
                  for(i = 0 ; i < ndomzs[i1]/2 ; i ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
                  
               }
               if(j1 < ndomy - 1 || ldy < dy - 1)
               {
                  /* one on the back, mark lower */
                  for(i = ndomzs[i1]/2 ; i < ndomzs[i1] ; i ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + i * numxs[ldx] * numys[ldy] + (ndomys[j1]-1) * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(i1 > 0 || ldz > 0)
               {
                  /* one down, mark upper */
                  for(j = 0 ; j < ndomys[j1]/2 ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               if(i1 < ndomz - 1 || ldz < dz - 1)
               {
                  /* one up, mark lower */
                  for(j = ndomys[j1]/2 ; j < ndomys[j1] ; j ++)
                  {
                     for(k = 0 ; k < ndomxs[k1] ; k ++)
                     {
                        ii = globalndomdisp[color] + (ndomzs[i1] - 1) * numxs[ldx] * numys[ldy] + j * numxs[ldx] + k;
                        this->_separator_domi[ii] = -1;
                     }
                  }
               }
               color++;
            }
         }
      } 
      
      numxs.Clear();
      dispxs.Clear();
      numys.Clear();
      dispys.Clear();
      numzs.Clear();
      dispzs.Clear();
      dispglobal.Clear();
      
      ndomxs.Clear();
      ndomxdisps.Clear();
      ndomys.Clear();
      ndomydisps.Clear();
      ndomzs.Clear();
      ndomzdisps.Clear();
      
      globalndomdisp.Clear();
      
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<complexs>::HelmholtzWithPartition(int n, int dx, int dy, int dz, int d2x, int d2y, int d2z, complexs w, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::HelmholtzWithPartition(int n, int dx, int dy, int dz, int d2x, int d2y, int d2z, complexd w, parallel_log &parlog);
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog)
   {
      /* TODO: this is a sequential read and not support long int */
      
      int                     err = 0, i, row, col;
      T                       val;
      int                     nrow_global = 0, ncol_global = 0, nnz_global = 0, nrow_local = 0, nrow_extra, ncol_local;
      int                     init_guess, offd_init_guess, domr, domc;
      vector_int              nrow_locals, nrow_disps, ncol_locals, ncol_disps;
      
      MPI_Comm comm;
      int np, myid;
      parlog.GetMpiInfo(np, myid, comm);
      
      CooMatrixClass<T>                global_coo_mat;
      CooMatrixClass<T>                local_diag_coo_mat;
      CooMatrixClass<T>                local_offd_coo_mat;
      std::vector<CooMatrixClass<T> >  diag_coo_mats;
      std::vector<CooMatrixClass<T> >  offd_coo_mats;
      std::vector<vector_long>         offd_maps;
      vector_int                       info;
      std::vector<vector_int>          offd_markers;
      
      if(np == 1)
      {
         err = global_coo_mat.ReadFromMMFile( matfile, idxin); PARGEMSLR_CHKERR(err);
         
         nrow_global = global_coo_mat.GetNumRowsLocal();
         ncol_global = global_coo_mat.GetNumColsLocal();
         
         this->Setup(nrow_global, 0, nrow_global, ncol_global, 0, ncol_global, parlog);
         
         global_coo_mat.ToCsr(this->GetDataLocation(), this->_diag_mat);
         this->_offd_mat.Setup(global_coo_mat.GetNumRowsLocal(), 0, 0);
         this->_offd_mat.GetIVector().Fill(0);
         
         return PARGEMSLR_SUCCESS;
      }
      
      if(myid == 0)
      {
         /* read the global matrix */
         err = global_coo_mat.ReadFromMMFile( matfile, idxin); PARGEMSLR_CHKERR(err);
         
         nrow_locals.Setup(np);
         ncol_locals.Setup(np);
         nrow_disps.Setup(np+1);
         ncol_disps.Setup(np+1);
         
         /* now we need to assign rows to different processors */
         nrow_global = global_coo_mat.GetNumRowsLocal();
         ncol_global = global_coo_mat.GetNumColsLocal();
         nnz_global = global_coo_mat.GetNumNonzeros();
         int *COO_i = global_coo_mat.GetI();
         int *COO_j = global_coo_mat.GetJ();
         T *COO_data = global_coo_mat.GetData();
         
         nrow_local = nrow_global / np;
         nrow_extra = nrow_global % np;
         ncol_local = ncol_global / np;
         
         nrow_disps[0] = 0;
         ncol_disps[0] = 0;
         for(i = 0 ; i < nrow_extra ; i ++)
         {
            nrow_locals[i] = nrow_local + 1;
            ncol_locals[i] = ncol_local + 1;
         }
         for(i = nrow_extra ; i < np ; i ++)
         {
            nrow_locals[i] = nrow_local;
            ncol_locals[i] = ncol_local;
         }
         for(i = 0 ; i < np ; i ++)
         {
            nrow_disps[i+1] = nrow_disps[i] + nrow_locals[i];
            ncol_disps[i+1] = ncol_disps[i] + ncol_locals[i];
         }
         
         diag_coo_mats.resize(np);
         offd_coo_mats.resize(np);
         offd_markers.resize(np);
         offd_maps.resize(np);
         
         init_guess = nnz_global/np;
         
         for(i = 0 ; i < np ; i ++)
         {
            offd_markers[i].Setup(ncol_global);
            offd_markers[i].Fill(-1);
            diag_coo_mats[i].Setup(nrow_locals[i], ncol_locals[i], init_guess);
            offd_init_guess = PargemslrMax(nrow_locals[i], ncol_locals[i]);
            offd_coo_mats[i].Setup(nrow_locals[i], INT_MAX, offd_init_guess);
            offd_maps[i].Setup( 0, offd_init_guess, kMemoryHost, false);
         }
         
         for(i = 0 ; i < nnz_global ; i ++)
         {
            row = COO_i[i];
            col = COO_j[i];
            val = COO_data[i];
            
            if( nrow_disps.BinarySearch(row, domr, true) < 0 )
            {
               /* if not found, is the location we insert, thus, we need to -1 for the domain */
               domr--;
            }
            
            if( ncol_disps.BinarySearch(col, domc, true) < 0 )
            {
               /* if not found, is the location we insert, thus, we need to -1 for the domain */
               domc--;
            }
            
            if(domr == domc)
            {
               /* diagonal */
               diag_coo_mats[domr].PushBack( row - nrow_disps[domr], col - ncol_disps[domc], val);
            }
            else
            {
               /* offdiagonal */
               if(offd_markers[domr][col] >= 0)
               {
                  /* already have this */
                  offd_coo_mats[domr].PushBack( row - nrow_disps[domr], offd_markers[domr][col], val);
               }
               else
               {
                  offd_markers[domr][col] = offd_maps[domr].GetLengthLocal();
                  offd_coo_mats[domr].PushBack( row - nrow_disps[domr], offd_markers[domr][col], val);
                  offd_maps[domr].PushBack(col);
               }
            }
         }
         
         for(i = 0 ; i < np ; i ++)
         {
            diag_coo_mats[i].SetNumNonzeros();
            offd_coo_mats[i].SetNumNonzeros();
            offd_coo_mats[i].SetNumCols(offd_maps[i].GetLengthLocal());
         }
         
      }
      
      /* now, info ready on rank 0, prepare to send */
      info.Setup(9);
      if(myid == 0)
      {
         info[4] = nrow_global;
         info[5] = ncol_global;
         for(i = 1 ; i < np ; i ++)
         {
            info[0] = nrow_locals[i];
            info[1] = ncol_locals[i];
            info[2] = nrow_disps[i];
            info[3] = ncol_disps[i];
            info[6] = diag_coo_mats[i].GetNumNonzeros();
            info[7] = offd_coo_mats[i].GetNumNonzeros();
            info[8] = offd_maps[i].GetLengthLocal();
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( info.GetData(), 9, i, 0, comm) );
         }
         
         /* MPI_Comm_dup will block! */
         this->Setup( nrow_locals[0], nrow_disps[0], nrow_global, ncol_locals[0], ncol_disps[0], ncol_global, parlog);
         
         for(i = 1 ; i < np ; i ++)
         {
            info[0] = nrow_locals[i];
            info[1] = ncol_locals[i];
            info[2] = nrow_disps[i];
            info[3] = ncol_disps[i];
            info[6] = diag_coo_mats[i].GetNumNonzeros();
            info[7] = offd_coo_mats[i].GetNumNonzeros();
            info[8] = offd_maps[i].GetLengthLocal();
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetI(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetJ(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetData(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetI(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetJ(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetData(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_maps[i].GetData(), info[8], i, 0, comm) );
         }
         
         this->_offd_map_v.Setup(offd_maps[0].GetLengthLocal());
         
         PARGEMSLR_MEMCPY( this->_offd_map_v.GetData(), offd_maps[0].GetData(), offd_maps[0].GetLengthLocal(), kMemoryHost, kMemoryHost, long int);
         
         diag_coo_mats[0].ToCsr(this->GetDataLocation(), this->_diag_mat);
         offd_coo_mats[0].ToCsr(this->GetDataLocation(), this->_offd_mat);
         this->SortOffdMap();
         
         for(i = 0 ; i < np ; i ++)
         {
            offd_markers[i].Clear();
            diag_coo_mats[i].Clear();
            offd_coo_mats[i].Clear();
            offd_maps[i].Clear();
         }
         
         std::vector<CooMatrixClass<T> >().swap(diag_coo_mats);
         std::vector<CooMatrixClass<T> >().swap(offd_coo_mats);
         std::vector<vector_long>().swap(offd_maps);
         std::vector<vector_int>().swap(offd_markers);
      }
      else
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( info.GetData(), 9, 0, 0, comm, MPI_STATUS_IGNORE) );
         
         this->Setup( info[0], info[2], info[4], info[1], info[3], info[5], parlog);
         local_diag_coo_mat.Setup( info[0], info[1], info[6], kMemoryHost);
         local_offd_coo_mat.Setup( info[0], info[8], info[7], kMemoryHost);
         this->_offd_map_v.Setup(info[8]);
         
         local_diag_coo_mat.GetIVector().Resize( info[6], false, false);
         local_diag_coo_mat.GetJVector().Resize( info[6], false, false);
         local_diag_coo_mat.GetDataVector().Resize( info[6], false, false);
         local_offd_coo_mat.GetIVector().Resize( info[7], false, false);
         local_offd_coo_mat.GetJVector().Resize( info[7], false, false);
         local_offd_coo_mat.GetDataVector().Resize( info[7], false, false);
         
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetI(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetJ(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetData(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetI(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetJ(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetData(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( this->_offd_map_v.GetData(), info[8], 0, 0, comm, MPI_STATUS_IGNORE) );
         
         local_diag_coo_mat.SetNumNonzeros();
         local_diag_coo_mat.ToCsr(this->GetDataLocation(), this->_diag_mat);
         local_offd_coo_mat.SetNumNonzeros();
         local_offd_coo_mat.ToCsr(this->GetDataLocation(), this->_offd_mat);
         this->SortOffdMap();
         
         local_diag_coo_mat.Clear();
         local_offd_coo_mat.Clear();
      }
      
      info.Clear();
      
      return err;
   }
   template int ParallelCsrMatrixClass<float>::ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog);
   template int ParallelCsrMatrixClass<double>::ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexs>::ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog);

   template <typename T>
   int ParallelCsrMatrixClass<T>::ReadFromSingleCSR(int n, int idxin, int *A_i, int *A_j, T *A_data, parallel_log &parlog)
   {
      /* TODO: this is a sequential read and not support long int */
      
      int                     err = 0, i, j1, j2, j, row, col;
      T                       val;
      int                     nrow_global = 0, ncol_global = 0, nnz_global = 0, nrow_local = 0, nrow_extra, ncol_local;
      int                     init_guess, offd_init_guess, domr, domc;
      vector_int              nrow_locals, nrow_disps, ncol_locals, ncol_disps;
      
      MPI_Comm comm;
      int np, myid;
      parlog.GetMpiInfo(np, myid, comm);
      
      CooMatrixClass<T>                global_coo_mat;
      CooMatrixClass<T>                local_diag_coo_mat;
      CooMatrixClass<T>                local_offd_coo_mat;
      std::vector<CooMatrixClass<T> >  diag_coo_mats;
      std::vector<CooMatrixClass<T> >  offd_coo_mats;
      std::vector<vector_long>         offd_maps;
      vector_int                       info;
      std::vector<vector_int>          offd_markers;
      
      /* conver into COO */
      global_coo_mat.Setup( n, n, nnz_global, kMemoryHost);
      for(i = 0 ; i < n ; i ++)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            global_coo_mat.PushBack( i, A_j[j]-idxin, A_data[j]);
         }
      }
      
      if(np == 1)
      {
         nrow_global = global_coo_mat.GetNumRowsLocal();
         ncol_global = global_coo_mat.GetNumColsLocal();
         
         this->Setup(nrow_global, 0, nrow_global, ncol_global, 0, ncol_global, parlog);
         
         global_coo_mat.ToCsr(this->GetDataLocation(), this->_diag_mat);
         this->_offd_mat.Setup(global_coo_mat.GetNumRowsLocal(), 0, 0);
         this->_offd_mat.GetIVector().Fill(0);
         
         return PARGEMSLR_SUCCESS;
      }
      
      if(myid == 0)
      {
         nrow_locals.Setup(np);
         ncol_locals.Setup(np);
         nrow_disps.Setup(np+1);
         ncol_disps.Setup(np+1);
         
         /* now we need to assign rows to different processors */
         nrow_global = global_coo_mat.GetNumRowsLocal();
         ncol_global = global_coo_mat.GetNumColsLocal();
         nnz_global = global_coo_mat.GetNumNonzeros();
         int *COO_i = global_coo_mat.GetI();
         int *COO_j = global_coo_mat.GetJ();
         T *COO_data = global_coo_mat.GetData();
         
         nrow_local = nrow_global / np;
         nrow_extra = nrow_global % np;
         ncol_local = ncol_global / np;
         
         nrow_disps[0] = 0;
         ncol_disps[0] = 0;
         for(i = 0 ; i < nrow_extra ; i ++)
         {
            nrow_locals[i] = nrow_local + 1;
            ncol_locals[i] = ncol_local + 1;
         }
         for(i = nrow_extra ; i < np ; i ++)
         {
            nrow_locals[i] = nrow_local;
            ncol_locals[i] = ncol_local;
         }
         for(i = 0 ; i < np ; i ++)
         {
            nrow_disps[i+1] = nrow_disps[i] + nrow_locals[i];
            ncol_disps[i+1] = ncol_disps[i] + ncol_locals[i];
         }
         
         diag_coo_mats.resize(np);
         offd_coo_mats.resize(np);
         offd_markers.resize(np);
         offd_maps.resize(np);
         
         init_guess = nnz_global/np;
         
         for(i = 0 ; i < np ; i ++)
         {
            offd_markers[i].Setup(ncol_global);
            offd_markers[i].Fill(-1);
            diag_coo_mats[i].Setup(nrow_locals[i], ncol_locals[i], init_guess);
            offd_init_guess = PargemslrMax(nrow_locals[i], ncol_locals[i]);
            offd_coo_mats[i].Setup(nrow_locals[i], INT_MAX, offd_init_guess);
            offd_maps[i].Setup( 0, offd_init_guess, kMemoryHost, false);
         }
         
         for(i = 0 ; i < nnz_global ; i ++)
         {
            row = COO_i[i];
            col = COO_j[i];
            val = COO_data[i];
            
            if( nrow_disps.BinarySearch(row, domr, true) < 0 )
            {
               /* if not found, is the location we insert, thus, we need to -1 for the domain */
               domr--;
            }
            
            if( ncol_disps.BinarySearch(col, domc, true) < 0 )
            {
               /* if not found, is the location we insert, thus, we need to -1 for the domain */
               domc--;
            }
            
            if(domr == domc)
            {
               /* diagonal */
               diag_coo_mats[domr].PushBack( row - nrow_disps[domr], col - ncol_disps[domc], val);
            }
            else
            {
               /* offdiagonal */
               if(offd_markers[domr][col] >= 0)
               {
                  /* already have this */
                  offd_coo_mats[domr].PushBack( row - nrow_disps[domr], offd_markers[domr][col], val);
               }
               else
               {
                  offd_markers[domr][col] = offd_maps[domr].GetLengthLocal();
                  offd_coo_mats[domr].PushBack( row - nrow_disps[domr], offd_markers[domr][col], val);
                  offd_maps[domr].PushBack(col);
               }
            }
         }
         
         for(i = 0 ; i < np ; i ++)
         {
            diag_coo_mats[i].SetNumNonzeros();
            offd_coo_mats[i].SetNumNonzeros();
            offd_coo_mats[i].SetNumCols(offd_maps[i].GetLengthLocal());
         }
         
      }
      
      /* now, info ready on rank 0, prepare to send */
      info.Setup(9);
      if(myid == 0)
      {
         info[4] = nrow_global;
         info[5] = ncol_global;
         for(i = 1 ; i < np ; i ++)
         {
            info[0] = nrow_locals[i];
            info[1] = ncol_locals[i];
            info[2] = nrow_disps[i];
            info[3] = ncol_disps[i];
            info[6] = diag_coo_mats[i].GetNumNonzeros();
            info[7] = offd_coo_mats[i].GetNumNonzeros();
            info[8] = offd_maps[i].GetLengthLocal();
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( info.GetData(), 9, i, 0, comm) );
         }
         
         /* MPI_Comm_dup will block! */
         this->Setup( nrow_locals[0], nrow_disps[0], nrow_global, ncol_locals[0], ncol_disps[0], ncol_global, parlog);
         
         for(i = 1 ; i < np ; i ++)
         {
            info[0] = nrow_locals[i];
            info[1] = ncol_locals[i];
            info[2] = nrow_disps[i];
            info[3] = ncol_disps[i];
            info[6] = diag_coo_mats[i].GetNumNonzeros();
            info[7] = offd_coo_mats[i].GetNumNonzeros();
            info[8] = offd_maps[i].GetLengthLocal();
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetI(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetJ(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( diag_coo_mats[i].GetData(), info[6], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetI(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetJ(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_coo_mats[i].GetData(), info[7], i, 0, comm) );
            PARGEMSLR_MPI_CALL( PargemslrMpiSend( offd_maps[i].GetData(), info[8], i, 0, comm) );
         }
         
         this->_offd_map_v.Setup(offd_maps[0].GetLengthLocal());
         
         PARGEMSLR_MEMCPY( this->_offd_map_v.GetData(), offd_maps[0].GetData(), offd_maps[0].GetLengthLocal(), kMemoryHost, kMemoryHost, long int);
         
         diag_coo_mats[0].ToCsr(this->GetDataLocation(), this->_diag_mat);
         offd_coo_mats[0].ToCsr(this->GetDataLocation(), this->_offd_mat);
         this->SortOffdMap();
         
         for(i = 0 ; i < np ; i ++)
         {
            offd_markers[i].Clear();
            diag_coo_mats[i].Clear();
            offd_coo_mats[i].Clear();
            offd_maps[i].Clear();
         }
         
         std::vector<CooMatrixClass<T> >().swap(diag_coo_mats);
         std::vector<CooMatrixClass<T> >().swap(offd_coo_mats);
         std::vector<vector_long>().swap(offd_maps);
         std::vector<vector_int>().swap(offd_markers);
      }
      else
      {
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( info.GetData(), 9, 0, 0, comm, MPI_STATUS_IGNORE) );
         
         this->Setup( info[0], info[2], info[4], info[1], info[3], info[5], parlog);
         local_diag_coo_mat.Setup( info[0], info[1], info[6], kMemoryHost);
         local_offd_coo_mat.Setup( info[0], info[8], info[7], kMemoryHost);
         this->_offd_map_v.Setup(info[8]);
         
         local_diag_coo_mat.GetIVector().Resize( info[6], false, false);
         local_diag_coo_mat.GetJVector().Resize( info[6], false, false);
         local_diag_coo_mat.GetDataVector().Resize( info[6], false, false);
         local_offd_coo_mat.GetIVector().Resize( info[7], false, false);
         local_offd_coo_mat.GetJVector().Resize( info[7], false, false);
         local_offd_coo_mat.GetDataVector().Resize( info[7], false, false);
         
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetI(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetJ(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_diag_coo_mat.GetData(), info[6], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetI(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetJ(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( local_offd_coo_mat.GetData(), info[7], 0, 0, comm, MPI_STATUS_IGNORE) );
         PARGEMSLR_MPI_CALL( PargemslrMpiRecv( this->_offd_map_v.GetData(), info[8], 0, 0, comm, MPI_STATUS_IGNORE) );
         
         local_diag_coo_mat.SetNumNonzeros();
         local_diag_coo_mat.ToCsr(this->GetDataLocation(), this->_diag_mat);
         local_offd_coo_mat.SetNumNonzeros();
         local_offd_coo_mat.ToCsr(this->GetDataLocation(), this->_offd_mat);
         this->SortOffdMap();
         
         local_diag_coo_mat.Clear();
         local_offd_coo_mat.Clear();
      }
      
      info.Clear();
      
      return err;
   }
   template int ParallelCsrMatrixClass<float>::ReadFromSingleCSR(int n, int idxin, int *A_i, int *A_j, float *A_data, parallel_log &parlog);
   template int ParallelCsrMatrixClass<double>::ReadFromSingleCSR(int n, int idxin, int *A_i, int *A_j, double *A_data,  parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexs>::ReadFromSingleCSR(int n, int idxin, int *A_i, int *A_j, complexs *A_data, parallel_log &parlog);
   template int ParallelCsrMatrixClass<complexd>::ReadFromSingleCSR(int n, int idxin, int *A_i, int *A_j, complexd *A_data, parallel_log &parlog);

   template <typename T>
   int ParallelCsrMatrixClass<T>::InsertGhostDiagonal()
   {
      int i, j1, j2, j, n_local, nnz_local, idx, missing, marker;
      int location = this->_diag_mat.GetDataLocation();
      
      if(location == kMemoryDevice)
      {
         this->_diag_mat.MoveData(kMemoryHost);
      }
      
      int *diag_i = this->_diag_mat.GetI();
      int *diag_j = this->_diag_mat.GetJ();
      T *diag_data = this->_diag_mat.GetData();
      
      n_local = this->_diag_mat.GetNumRowsLocal();
      nnz_local = this->_diag_mat.GetNumNonzeros();
      
      if(n_local == 0)
      {
         // nothing to do
         this->_diag_mat.MoveData(location);
         return PARGEMSLR_SUCCESS;
      }
      
      // check if any missing
      missing = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = diag_i[i];
         j2 = diag_i[i+1];
         marker = 0;
         for(j = j1 ; j < j2 ; j ++)
         {
            if(diag_j[j] == i)
            {
               marker = 1;
            }
         }
         if(marker == 0)
         {
            missing = 1;
            break;
         }
      }
      
      if(!missing)
      {
         this->_diag_mat.MoveData(location);
         return PARGEMSLR_SUCCESS;
      }
      
      // Count missing amount
      missing = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = diag_i[i];
         j2 = diag_i[i+1];
         marker = 0;
         for(j = j1 ; j < j2 ; j ++)
         {
            if(diag_j[j] == i)
            {
               marker = 1;
            }
         }
         if(marker == 0)
         {
            missing ++;
         }
      }
      
      // now create a new CSR matrix
      CsrMatrixClass<T> new_diag;
      new_diag.Setup( n_local, n_local, nnz_local + missing, true);
      
      idx = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = diag_i[i];
         j2 = diag_i[i+1];
         marker = 0;
         for(j = j1 ; j < j2 ; j ++)
         {
            if(diag_j[j] == i)
            {
               marker = 1;
            }
            new_diag.PushBack( diag_j[j], diag_data[j]);
            idx++;
         }
         if(marker == 0)
         {
            new_diag.PushBack( i, T());
            idx++;
         }
         new_diag.GetI()[i+1] = idx;
      }
      
      this->_diag_mat.Clear();
      this->_diag_mat = std::move(new_diag);
      
      this->_diag_mat.MoveData(location);
      return PARGEMSLR_SUCCESS;
      
   }
   template int ParallelCsrMatrixClass<float>::InsertGhostDiagonal();
   template int ParallelCsrMatrixClass<double>::InsertGhostDiagonal();
   template int ParallelCsrMatrixClass<complexs>::InsertGhostDiagonal();
   template int ParallelCsrMatrixClass<complexd>::InsertGhostDiagonal();
   
   template <typename T>
   int ParallelCsrMatrixClass<T>::PlotPatternGnuPlot( const char *datafilename, int pttype)
   {
      if(this->GetDataLocation() == kMemoryDevice)
      {
         PARGEMSLR_ERROR("Plot Pattern only works on host.");
      }
      
      MPI_Comm comm;
      int np, myid;
      this->GetMpiInfo(np, myid, comm);
      
      char filename[1024];
      snprintf( filename, 1024, "./TempData/%s%05d", datafilename, myid );
      
      int i, j, j1, j2;
   
      FILE *fdata, *pgnuplot;
      
      if ((fdata = fopen(filename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return PARGEMSLR_ERROR_IO_ERROR;
      }
      
      int *diag_i = this->_diag_mat.GetI();
      int *diag_j = this->_diag_mat.GetJ();
      int *offd_i = this->_offd_mat.GetI();
      int *offd_j = this->_offd_mat.GetJ();
      long int *offd_map = this->_offd_map_v.GetData();
      
      for(i = 0 ; i < _nrow_local ; i ++)
      {
         j1 = diag_i[i];
         j2 = diag_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            fprintf(fdata, "%ld %ld \n", _ncol_start+diag_j[j]+1, _nrow_global-_nrow_start-i);
         }
         j1 = offd_i[i];
         j2 = offd_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            fprintf(fdata, "%ld %ld \n", offd_map[offd_j[j]]+1, _nrow_global-_nrow_start-i);
         }
      }
      
      fclose(fdata);
      
      PARGEMSLR_MPI_CALL(MPI_Barrier(comm));
      
      long int nnz = this->GetNumNonzeros();
      
      if(myid == 0)
      {
         
         if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
         {
            printf("Can't open gnuplot file.\n");
            return PARGEMSLR_ERROR_IO_ERROR;
         }
         
         fprintf(pgnuplot, "set title \"nnz = %ld\"\n", nnz);
         fprintf(pgnuplot, "set xrange [0:%ld]\n", _ncol_global+1);
         fprintf(pgnuplot, "set yrange [0:%ld]\n", _nrow_global+1);
         fprintf(pgnuplot, "plot '%s' pt %d", filename, pttype);
         for(i = 1 ; i < np ; i ++)
         {
            char tempfilename[1024];
            snprintf( tempfilename, 1024, "./TempData/%s%05d", datafilename, i );
            fprintf(pgnuplot, ", '%s' pt %d", tempfilename, pttype);
         }
         fprintf(pgnuplot, "\n");
         pclose(pgnuplot);
      }
   
      return PARGEMSLR_SUCCESS;
   }
   template int ParallelCsrMatrixClass<float>::PlotPatternGnuPlot( const char *datafilename, int pttype);
   template int ParallelCsrMatrixClass<double>::PlotPatternGnuPlot( const char *datafilename, int pttype);
   template int ParallelCsrMatrixClass<complexs>::PlotPatternGnuPlot( const char *datafilename, int pttype);
   template int ParallelCsrMatrixClass<complexd>::PlotPatternGnuPlot( const char *datafilename, int pttype);
   
}
