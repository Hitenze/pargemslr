#ifndef PARGEMSLR_PARALLEL_CSR_MATRIX_H
#define PARGEMSLR_PARALLEL_CSR_MATRIX_H

/**
 * @file parallel_csr_matrix.hpp
 * @brief Parallel csr matrix data structure.
 */
#include <iostream>

#include "../utils/utils.hpp"
#include "../utils/structs.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"

#ifdef PARGEMSLR_CUDA
#include "cusparse.h"
#endif

namespace pargemslr
{
   
   /** 
    * @brief   The data structure for parallel matvec helper, store communication inforamtion for parallel matvec, and working buffers.
    * @details The data structure for parallel matvec helper, store communication inforamtion for parallel matvec, and working buffers.
    */
   class CommunicationHelperClass: public parallel_log
   {
   private:
      
      /**
       * @brief   The location of send idx data.
       * @details The location of send idx data.
       */
      int                                 _location_send;
      
      /**
       * @brief   The location of recv idx data.
       * @details The location of recv idx data.
       */
      int                                 _location_recv;
      
      /** 
       * @brief   The size of the working vector in bypes.
       * @details The size of the working vector in bypes.
       */
      int                                 _buff_unit_size;
      
      /** 
       * @brief   Working buffer used in communication.
       * @details Working buffer used in communication.
       */
      std::vector<void*>                  _send_buff_v2;
      
      /** 
       * @brief   Working buffer used in communication.
       * @details Working buffer used in communication.
       */
      std::vector<void*>                  _recv_buff_v2;

#ifdef PARGEMSLR_CUDA

      /** 
       * @brief   The size of the working vector in bypes.
       * @details The size of the working vector in bypes.
       */
      int                                 _buff_d_unit_size;
      
      /** 
       * @brief   Working device buffer used in communication.
       * @details Working device buffer used in communication.
       */
      std::vector<void*>                  _send_buff_v2_d;
      
      /** 
       * @brief   Working device buffer used in communication.
       * @details Working device buffer used in communication.
       */
      std::vector<void*>                  _recv_buff_v2_d;

#endif

   public:
      
      /** 
       * @brief   The size of the input vector.
       * @details The size of the input vector.
       */
      int                                 _n_in;
      
      /** 
       * @brief   The size of the output vector.
       * @details The size of the output vector.
       */
      int                                 _n_out;
      
      /** 
       * @brief   The vector stores the MPI ranks that myid needs to send local vector to.
       * @details The vector stores the MPI ranks that myid needs to send local vector to.
       */
      vector_int                          _send_to_v;
      
      /** 
       * @brief   The vector stores the MPI ranks that myid needs to recv external vector from.
       * @details The vector stores the MPI ranks that myid needs to recv external vector from.
       */
      vector_int                          _recv_from_v;
      
      /** 
       * @brief   2D vector, send_idx_v2[i] is the index in local vector that need to be sent to proc. i.
       * @details 2D vector, send_idx_v2[i] is the index in local vector that need to be sent to proc. i.
       */
      std::vector<vector_int >            _send_idx_v2;
      
      /** 
       * @brief   2D vector, recv_idx_v2[i] is the index in external vector that need to be received from proc. i.
       * @details 2D vector, recv_idx_v2[i] is the index in external vector that need to be received from proc. i. \n
       *          The external vector can be directly used in matvec with offd_mat of trans(offd_mat) depending on the type of the current matvec helper.
       */
      std::vector<vector_int >            _recv_idx_v2;
      
      /** 
       * @brief   Helper array for building the comm helper.
       * @details Helper array for building the comm helper.
       */
      std::vector<vector_int >            _idx_helper_v2;
      
      /** 
       * @brief   Vector of MPI_Request for communication.
       * @details Vector of MPI_Request for communication.
       */
      vector<MPI_Request>                 _requests_v;
      
      /** 
       * @brief   Is true when the comm_helper ready to use.
       * @details Is true when the comm_helper ready to use.
       */
      bool                                _is_ready;
      
      /** 
       * @brief   Is true when there is communication on going.
       * @details Is true when there is communication on going.
       */
      bool                                _is_waiting;
      
      /** 
       * @brief   If the neiborhood information (send_to and recv_from) is set in advance.
       * @details If the neiborhood information (send_to and recv_from) is set in advance.
       */
      bool                                _is_nbhd_built;
      
      /**
       * @brief   The constructor of CommunicationHelperClass.
       * @details The constructor of CommunicationHelperClass.
       */
      CommunicationHelperClass();
      
      /**
       * @brief   The copy constructor of CommunicationHelperClass.
       * @details The copy constructor of CommunicationHelperClass.
       * @param   [in] comm_helper The target CommunicationHelperClass.
       */
      CommunicationHelperClass(const CommunicationHelperClass &comm_helper);
      
      /**
       * @brief   The move constructor of CommunicationHelperClass.
       * @details The move constructor of CommunicationHelperClass.
       * @param   [in] comm_helper The target CommunicationHelperClass.
       */
      CommunicationHelperClass(CommunicationHelperClass &&comm_helper);
      
      /**
       * @brief   The = operator of CommunicationHelperClass.
       * @details The = operator of CommunicationHelperClass.
       * @param   [in] comm_helper The target CommunicationHelperClass.
       * @return     Return the CommunicationHelperClass.
       */
      CommunicationHelperClass& operator= (const CommunicationHelperClass &comm_helper);
      
      /**
       * @brief   The = operator of CommunicationHelperClass.
       * @details The = operator of CommunicationHelperClass.
       * @param   [in] comm_helper The target CommunicationHelperClass.
       * @return     Return the CommunicationHelperClass.
       */
      CommunicationHelperClass& operator= (CommunicationHelperClass &&comm_helper);
      
      /**
       * @brief   The destructor of CommunicationHelperClass.
       * @details The destructor of CommunicationHelperClass, just a call to the free function.
       */
      ~CommunicationHelperClass();
      
      /**
       * @brief   Free the current matvec helper, set everything to 0.
       * @details Free the current matvec helper, set everything to 0.
       * @return           return Error message.
       */
      int                           Clear();
      
      /**
       * @brief   Create buffer.
       * @details Create buffer.
       * @param [in]  unitsize The unit size in byte.
       * @return           return Error message.
       */
      int                           CreateHostBuffer(int unitsize);

#ifdef PARGEMSLR_CUDA

      /**
       * @brief   Create buffer on device.
       * @details Create buffer on device.
       * @param [in]  unitsize The unit size in byte.
       * @return           return Error message.
       */
      int                           CreateDeviceBuffer(int unitsize);

#endif

      /**
       * @brief   Move the index data to other location in the memory.
       * @details Move the index data to other location in the memory.
       * @return           Return error message.
       */
      int                           MoveData(int location);
      
      /**
       * @brief   Move the send data to other location in the memory.
       * @details Move the send data to other location in the memory.
       * @return           Return error message.
       */
      int                           MoveSendData(int location);
      
      /**
       * @brief   Move the recv data to other location in the memory.
       * @details Move the recv data to other location in the memory.
       * @return           Return error message.
       */
      int                           MoveRecvData(int location);
      
      /**
       * @brief   Apply communication using the buffer inside this comm helper.
       * @details Apply communication using the buffer inside this comm helper.
       * @param [in]  vec_in The vector we send information from.
       * @param [out] vec_out The vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_in The data location of the input data.
       * @param [in]  loc_out The data location of the output data.
       * @return     Return error information.
       */
      template <typename T>
      int                           DataTransfer(const VectorVirtualClass<T> &vec_in, VectorVirtualClass<T> &vec_out, int loc_in, int loc_out);
      
      /**
       * @brief   Start communication using the buffer inside this comm helper.
       * @details Start communication using the buffer inside this comm helper.
       * @note    Need to call DataTransferOver function to make sure the communication finished.
       * @param [in] vec_in The vector we send information from.
       * @param [in]  loc_in The data location of the input data.
       * @return     Return error information.
       */
      template <typename T>
      int                           DataTransferStart(const VectorVirtualClass<T> &vec_in, int loc_in);
      
      /**
       * @brief   Finish communication using the buffer inside this comm helper.
       * @details Finish communication using the buffer inside this comm helper.
       * @param [out] vec_out The vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_out The data location of the output data.
       * @return      Return error information.
       */
      template <typename T>
      int                           DataTransferOver(VectorVirtualClass<T> &vec_out, int loc_out);
      
      /**
       * @brief   Apply communication using the buffer inside this comm helper.
       * @details Apply communication using the buffer inside this comm helper.
       * @param [in]  vec_in The 2D vector we send information from.
       * @param [out] vec_out The 2D vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_in The data location of the input data. For 2D array, this is the second dimension. The fist dimention std::vector is always on the host.
       * @param [in]  loc_out The data location of the output data. For 2D array, this is the second dimension. The fist dimention std::vector is always on the host.
       * @return     Return error information.
       */
      template <typename T, class VectorType>
      int                           DataTransfer(const std::vector<VectorType> &vec_in, std::vector<VectorType> &vec_out, int loc_in, int loc_out);
      
      /**
       * @brief   Start communication using the buffer inside this comm helper.
       * @details Start communication using the buffer inside this comm helper.
       * @note    Need to call DataTransferOver function to make sure the communication finished.
       * @param [in] vec_in The 2D vector we send information from.
       * @param [in] loc_in The data location of the input data. For 2D array, this is the second dimension. The fist dimention std::vector is always on the host.
       * @return     Return error information.
       */
      template <typename T, class VectorType>
      int                           DataTransferStart(const std::vector<VectorType> &vec_in, int loc_in);
      
      /**
       * @brief   Finish communication using the buffer inside this comm helper.
       * @details Finish communication using the buffer inside this comm helper.
       * @param [out] vec_out The 2D vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_out The data location of the output data. For 2D array, this is the second dimension. The fist dimention std::vector is always on the host.
       * @return      Return error information.
       */
      template <typename T, class VectorType>
      int                           DataTransferOver(std::vector<VectorType> &vec_out, int loc_out);
      
      /**
       * @brief   Apply communication using the buffer inside this comm helper in the opposite direction.
       * @details Apply communication using the buffer inside this comm helper in the opposite direction.
       * @param [in]  vec_in The vector we send information from.
       * @param [out] vec_out The vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_in The data location of the input data.
       * @param [in]  loc_out The data location of the output data.
       * @return     Return error information.
       */
      template <typename T>
      int                           DataTransferReverse(const VectorVirtualClass<T> &vec_in, VectorVirtualClass<T> &vec_out, int loc_in, int loc_out);
      
      /**
       * @brief   Start communication using the buffer inside this comm helper in the opposite direction.
       * @details Start communication using the buffer inside this comm helper in the opposite direction.
       * @note    Need to call DataTransferOver function to make sure the communication finished.
       * @param [in] vec_in The vector we send information from.
       * @param [in]  loc_in The data location of the input data.
       * @return     Return error information.
       */
      template <typename T>
      int                           DataTransferStartReverse(const VectorVirtualClass<T> &vec_in, int loc_in);
      
      /**
       * @brief   Finish communication using the buffer inside this comm helper in the opposite direction.
       * @details Finish communication using the buffer inside this comm helper in the opposite direction.
       * @param [out] vec_out The vector we recv information to. Space should be reserved before calling this function.
       * @param [in]  loc_out The data location of the output data.
       * @return      Return error information.
       */
      template <typename T>
      int                           DataTransferOverReverse(VectorVirtualClass<T> &vec_out, int loc_out);
      
   };
   
	/**
    * @brief   Class of parallel CSR matrices.
    * @details Class of parallel CSR matrices.
    */
   template <typename T>
   class ParallelCsrMatrixClass: public MatrixClass<T>
   {
   private:
      
      /* variables */
      
      /**
       * @brief   Global number of rows.
       * @details Global number of rows.
       */
      long int                   _nrow_global;
      
      /**
       * @brief   Global number of cols.
       * @details Global number of cols.
       */
      long int                   _ncol_global;
      
      /**
       * @brief   Start index of rows.
       * @details Start index of rows.
       */
      long int                   _nrow_start;
      
      /**
       * @brief   Start index of cols.
       * @details Start index of cols.
       */
      long int                   _ncol_start;
      
      /**
       * @brief   Local number of rows.
       * @details Local number of rows.
       */
      int                        _nrow_local;
      
      /**
       * @brief   Local number of cols.
       * @details Local number of cols.
       */
      int                        _ncol_local;
      
      /** 
       * @brief   Set to true if the _offd_map_v is sorted in ascending order.
       * @details Set to true if the _offd_map_v is sorted in ascending order.
       * @note    Be sure to change this into false if you change the offd_map_v in your own code.
       */
      bool                       _is_offd_map_sorted;
      
      /** 
       * @brief   The diagonal matrix in CSR format. Diagonal cols are local.
       * @details The diagonal matrix in CSR format. Diagonal cols are local.
       */
      CsrMatrixClass<T>          _diag_mat;
      
      /** 
       * @brief   The offdiagonal matrix in CSR format. Offdiagonal means cols are exterior.
       * @details The offdiagonal matrix in CSR format. Offdiagonal means cols are exterior. \n
       *          The column numbers here are not the global column number, need to map with offd_map_v.
       */
      CsrMatrixClass<T>          _offd_mat;
      
      /** 
       * @brief   Map from column number in offd_mat to the exact global column number.
       * @details Map from column number in offd_mat to the exact global column number. This is useful in parallel matvec.
       */
      vector_long                _offd_map_v;
      
      /** 
       * @brief   Working vector for matvec.
       * @details Working vector for matvec.
       */
      SequentialVectorClass<T>   _matvec_working_vec;
      
      /** 
       * @brief   Working vector for matvec.
       * @details Working vector for matvec.
       */
      SequentialVectorClass<T>   _trans_matvec_working_vec;
      
      /** 
       * @brief   Number of pre assigned subdomains.
       * @details Number of pre assigned subdomains.
       */
      int                        _separator_ndom;
      
      /** 
       * @brief   Number of subdomain for each node, -1 is the separator.
       * @details Number of subdomain for each node, -1 is the separator.
       */
      vector_int                 _separator_domi;
      
   public:
      
      /**
       * @brief   The diagonal complex shift in some preconditioners. The preconditioner would be built on A+_diagonal_shift*I.
       * @details The diagonal complex shift in some preconditioners. The preconditioner would be built on A+_diagonal_shift*I.
       * @note    TODO: if useful, change to private.
       */
      typename std::conditional<PargemslrIsDoublePrecision<T>::value, 
                                 double, 
                                 float>::type _diagonal_shift;
      
      /** 
       * @brief   The data struxture for the parallel matvec. Stores communication inforamtion, and working buffers.
       * @details The data struxture for the parallel matvec. Stores communication inforamtion, and working buffers.
       */
      CommunicationHelperClass   _comm_helper;
         
      /** 
       * @brief   The data struxture for the parallel matvec of A transpose. Stores communication inforamtion, and working buffers.
       * @details The data struxture for the parallel matvec of A transpose. Stores communication inforamtion, and working buffers.
       * @note    Not yet in use.
       */
      CommunicationHelperClass   _trans_comm_helper;
      
      /**
       * @brief   The constructor of ParallelCsrMatrixClass.
       * @details The constructor of ParallelCsrMatrixClass. The default memory location is the host memory.
       */
      ParallelCsrMatrixClass();
      
      /**
       * @brief   The copy constructor of ParallelCsrMatrixClass.
       * @details The copy constructor of ParallelCsrMatrixClass.
       * @param   [in] mat The target matrix.
       */
      ParallelCsrMatrixClass(const ParallelCsrMatrixClass<T> &mat);
      
      /**
       * @brief   The move constructor of ParallelCsrMatrixClass.
       * @details The move constructor of ParallelCsrMatrixClass.
       * @param   [in] mat The target matrix.
       */
      ParallelCsrMatrixClass(ParallelCsrMatrixClass<T> &&mat);
      
      /**
       * @brief   The = operator of ParallelCsrMatrixClass.
       * @details The = operator of ParallelCsrMatrixClass.
       * @param   [in] mat The target matrix.
       * @return     Return the matrix.
       */
      ParallelCsrMatrixClass<T>& operator= (const ParallelCsrMatrixClass<T> &mat);
      
      /**
       * @brief   The = operator of ParallelCsrMatrixClass.
       * @details The = operator of ParallelCsrMatrixClass.
       * @param   [in] mat The target matrix.
       * @return     Return the matrix.
       */
      ParallelCsrMatrixClass<T>& operator= (ParallelCsrMatrixClass<T> &&mat);
      
      /**
       * @brief   The destructor of ParallelCsrMatrixClass.
       * @details The destructor of ParallelCsrMatrixClass. Simply a call to the free function.
       */
      virtual ~ParallelCsrMatrixClass();
      
      /**
       * @brief   Free the current parallel matrix, and set up new paralle matrix.
       * @details Free the current parallel matrix, and set up new paralle matrix. 
       * @note    The diag_mat and offd_mat need to be initialized separately.
       * @param   [in] nrow_local The local number of rows.
       * @param   [in] ncol_local The local number of cols.
       * @param   [in] parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int nrow_local, int ncol_local, parallel_log &parlog);
      
      /**
       * @brief   Free the current parallel matrix, and set up new paralle matrix. The default nnz is 0.
       * @details Free the current parallel matrix, and set up new paralle matrix. The default nnz is 0.
       * @note    The diag_mat and offd_mat need to be initialized separately.
       * @param   [in] nrow_local The local number of rows.
       * @param   [in] nrow_start The start index of the rows.
       * @param   [in] nrow_global The global length of the rows.
       * @param   [in] ncol_local The local number of cols.
       * @param   [in] ncol_start The start index of the columns.
       * @param   [in] ncol_global The global length of the columns.
       * @param   [in] parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Setup(int nrow_local, long int nrow_start, long int nrow_global, int ncol_local, long int ncol_start, long int ncol_global, parallel_log &parlog);
      
      /**
       * @brief   Free the current matrix.
       * @details Free the current matrix.
       * @return  Return error message.
       */
      virtual int    Clear();
      
      /**
       * @brief   Get the data location of the matrix.
       * @details Get the data location of the matrix.
       * @return     Return the locatino of the matrix.
       */
      virtual int    GetDataLocation() const;
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      virtual int    GetNumRowsLocal() const;
      
      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      virtual int    GetNumColsLocal() const;
      
      /**
       * @brief   Get the global number of rows of the matrix.
       * @details Get the global number of rows of the matrix.
       * @return     Return the global number of rows of the matrix.
       */
      long int       GetNumRowsGlobal() const;
      
      /**
       * @brief   Get the global number of columns of the matrix.
       * @details Get the global number of columns of the matrix.
       * @return     Return the global number of columns of the matrix.
       */
      long int       GetNumColsGlobal() const;
      
      /**
       * @brief   Get the global number of the first row of the matrix.
       * @details Get the global number of the first row of the matrix.
       * @return     Return the global number of the first row of the matrix.
       */
      long int       GetRowStartGlobal() const;
      
      /**
       * @brief   Get the global number of the first column of the matrix.
       * @details Get the global number of the first column of the matrix.
       * @return     Return the global number of the first column of the matrix.
       */
      long int       GetColStartGlobal() const;
      
      /**
       * @brief   Get the number of nonzeros in this matrix.
       * @details Get the number of nonzeros in this matrix.
       * @return     Return the number of nonzeros of the matrix.
       */
      virtual long int  GetNumNonzeros() const;
      
      /**
       * @brief   Get the vector arrays for the graph, used as inputs for ParMETIS. Diagonal entry is removed.
       * @details Get the vector arrays for the graph, used as inputs for ParMETIS. Diagonal entry is removed.
       * @param   [in] vtxdist The vertex distribution of size np+1.
       * @param   [in] xadj The local I in CSR format.
       * @param   [in] adjncy The local J in CSR format.
       * @return     Return error message.
       */
      int            GetGraphArrays( vector_long &vtxdist, vector_long &xadj, vector_long &adjncy);
      
      /**
       * @brief   Get the diagonal matrix.
       * @details Get the diagonal matrix.
       * @return  Return the diagonal matrix.
       */
      CsrMatrixClass<T>&            GetDiagMat();
      
      /**
       * @brief   Get the off-diagonal matrix.
       * @details Get the off-diagonal matrix.
       * @return  Return the off-diagonal matrix.
       */
      CsrMatrixClass<T>&            GetOffdMat();
      
      /**
       * @brief   Get the off-diagonal column map vector.
       * @details Get the off-diagonal column map vector.
       * @return  Return the off-diagonal column map vector.
       */
      vector_long&                  GetOffdMap();
      
      /**
       * @brief   Tell if the off-diagonal map vector is sorted.
       * @details Tell if the off-diagonal map vector is sorted.
       * @return  Return true if off-diagonal map is sorted.
       */
      bool&          IsOffdMapSorted();
      
      /**
       * @brief   Set the off-diagonal map vector to sorted/unsorted.
       * @details Set the off-diagonal map vector to sorted/unsorted.
       * @return  Return error message.
       */
      int            SetOffdMatSorted(bool is_offd_map_sorted);
      
      /**
       * @brief   Sort the off-diagonal map vector.
       * @details Sort the off-diagonal map vector.
       * @return     Return error message.
       */
      int            SortOffdMap();
      
      /**
       * @brief   Update the structure of a parallel vector to have same distribution as the row distribution of this matrix.
       * @details Update the structure of a parallel vector to have same distribution as the row distribution of this matrix.
       * @param [in,out]   The target vector.
       * @return     Return error message.
       */
      int            SetupVectorPtrStr(ParallelVectorClass<T> &vec);
      
      /**
       * @brief   Create an indentity matrix.
       * @details Create an indentity matrix. Should setup the diag_mat and offd_mat first.
       * @return     Return error message.
       */
      virtual int    Eye();
      
      /**
       * @brief   Fill the matrix with constant value.
       * @details Fill the matrix with constant value.
       * @param   [in]   v The value to be filled.
       * @return     Return error message.
       */
      virtual int    Fill(const T &v);
      
      /**
       * @brief      Move the data to another memory location.
       * @details    Move the data to another memory location.
       * @param [in] location The location move to.
       * @return     Return error message.
       */
      virtual int    MoveData( const int &location);
      
      /**
       * @brief      Scale the current csr matrix.
       * @details    Scale the current csr matrix.
       * @param [in] alpha The scale.
       * @return     Return error message.
       */
      virtual int    Scale(const T &alpha);
      
      /**
       * @brief         Get the value leads to max diagonal no.
       * @details       Scale the current csr matrix.
       * @param [in]    scale The scale.
       * @return        Return error message.
       */
      int            GetDiagScale(T &scale);
      
      /**
       * @brief      Transpost the current matrix.
       * @details    Transpost the current matrix.
       * @return     Return error message.
       */
      int            Transpose();
      
      
      /**
       * @brief      Transpost the current matrix.
       * @details    Transpost the current matrix.
       * @param [out]   AT The output matrix.
       * @return     Return error message.
       */
      int            Transpose(ParallelCsrMatrixClass<T> &AT);
      
      /**
       * @brief   Setup the matvec_helper for parallel matvec.
       * @details Setup the matvec_helper for parallel matvec.
       * @return  Return error message.
       */
      int            SetupMatvec();
      
      /**
       * @brief   Setup the matvec_helper for parallel matvec.
       * @details Setup the matvec_helper for parallel matvec. \n
                  Non-blocking communication, need to call SetupMatvecOver function to finish up.
       * @return  Return error message.
       */
      int            SetupMatvecStart();
      
      /**
       * @brief   Finishing up the setup phase of matvec_helper for parallel matvec.
       * @details Finishing up the setup phase of matvec_helper for parallel matvec.
       * @return  Return error message.
       */
      int            SetupMatvecOver();
      
      /**
       * @brief   In place parallel csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place parallel csr Matrix-Vector product ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @note    TODO: Currently A' is not supported.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      virtual int    MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y);
      
      /**
       * @brief   In place Matrix-Vector product ==>  z := alpha*A*x + beta*y, or z := alpha*A'*x + beta*y. Note that z should not equal to y.
       * @details In place Matrix-Vector product ==>  z := alpha*A*x + beta*y, or z := alpha*A'*x + beta*y. Note that z should not equal to y.
       * @note    TODO: Currently A' is not supported.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in]       y The second vector.
       * @param [out]      z The output vector.
       * @return           Return error message.
       */
      virtual int    MatVec( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, const VectorClass<T> &y, VectorClass<T> &z);
      
      /**
       * @brief   In place parallel csr Matrix-Vector product with offdiagonal block only ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place parallel csr Matrix-Vector product with offdiagonal block only ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The left vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The product vector.
       * @return           Return error message.
       */
      int            MatVecOffd( char trans, const T &alpha, const VectorClass<T> &x, const T &beta, VectorClass<T> &y);
      
      /**
       * @brief   Dense Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C where C is this matrix.
       * @details Dense Matrix-Matrix multiplication, C = alpha*transa(A)*transb(B)+beta*C where C is this matrix. \n
       *          If C is not pre-allocated, C will be allocated such that it has the same location as A.
       * @note    TODO: currently not supported.
       * @param [in]       alpha The alpha value.
       * @param [in]       A The first csr matrix.
       * @param [in]       transa Whether we transpose A or not.
       * @param [in]       B The second csr matrix.
       * @param [in]       transb Whether we transpose B or not.
       * @param [in]       beta The beta value.
       * @return           Return error message.
       */
      int            MatMat( const T &alpha, const ParallelCsrMatrixClass<T> &A, char transa, const ParallelCsrMatrixClass<T> &B, char transb, const T &beta);
      
      /**
       * @brief   Extract parallel submatrix from parallel CSR matrix.
       * @details Extract parallel submatrix from parallel CSR matrix. \n
       *          Each processor prvide several local rows/cols to keep (in global indices). No exchange of global information.
       * @param [in]    rows The rows to keep. The permutation in rows also applied.
       * @param [in]    cols The columns to keep. The permutation in cols also applied.
       * @param [in]    location The location of the output matrix.  
       * @param [out]   parcsrmat_out The output matrix.
       * @return           Return error message.                                                                                                                                                
       */
      int            SubMatrix(vector_long &rows, vector_long &cols, int location, ParallelCsrMatrixClass<T> &parcsrmat_out);
      
      /**
       * @brief   Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem.
       * @details Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem.
       * @param   [in]    nx
       * @param   [in]    ny
       * @param   [in]    nz Problem size on each dimension
       * @param   [in]    dx
       * @param   [in]    dy
       * @param   [in]    dz the decomposition of the problem on each dimension. Should have np == dx*dy*dz.
       * @param   [in]    alphax
       * @param   [in]    alphay
       * @param   [in]    alphaz Alpha value on each direction
       * @param   [in]    shift The shift value on diagonal
       * @param   [in]    parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one
       * @param   [in]    rand_perturb Set to true to add extra random real pertubation to diagonal entries as abs(shift)*Rand().
       *                  We only add shift that might makes the problem harder.
       * @return     Return error message.
       */
      int            Laplacian(int nx, int ny, int nz, int dx, int dy, int dz, T alphax, T alphay, T alphaz, T shift, parallel_log &parlog, bool rand_perturb = false);
      
      /**
       * @brief   Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f.
       * @details Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f. Robin boundary condition u'-iwu = 0 is used.
       * @param   [in]    n Problem size on each dimension
       * @param   [in]    dx
       * @param   [in]    dy
       * @param   [in]    dz the decomposition of the problem on the z dimension. Should have np == dx*dy*dz.
       * @param   [in]    w The w value.
       * @param   [in]    parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            Helmholtz(int n, int dx, int dy, int dz, T w, parallel_log &parlog);
      
      /**
       * @brief   Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem.
       * @details Generate Laplacian matrix on the host memory, 5-pt for 2D problem and 7-pt for 3D problem. \n
       *          Meth partition is created in this case based on infor in d2x/y/z. \n
       * @param   [in]    nx
       * @param   [in]    ny
       * @param   [in]    nz Problem size on each dimension
       * @param   [in]    dx
       * @param   [in]    dy
       * @param   [in]    dz the decomposition of the processors on each dimension. Should have dx*dy*dz == np.
       * @param   [in]    d2x
       * @param   [in]    d2y
       * @param   [in]    d2z the decomposition of the problem on each dimension. Should have d2x*d2y*d2z % np == 0.
       * @param   [in]    alphax
       * @param   [in]    alphay
       * @param   [in]    alphaz Alpha value on each direction
       * @param   [in]    shift The shift value on diagonal
       * @param   [in]    parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @param   [in]    rand_perturb Set to true to add extra random real pertubation to diagonal entries as abs(shift)*Rand().
       *                  We only add shift that might makes the problem harder.
       * @return     Return error message.
       */
      int            LaplacianWithPartition(int nx, int ny, int nz, int dx, int dy, int dz, int d2x, int d2y, int d2z, T alphax, T alphay, T alphaz, T shift, parallel_log &parlog, bool rand_perturb = false);
      
      /**
       * @brief   Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f.
       * @details Generate 3D Helmholtz matrix with 7-pt on [0,1]^3. -\Delta u - w^2 u = f. Robin boundary condition u'-iwu = 0 is used.
       * @param   [in]    n Problem size on each dimension
       * @param   [in]    dx
       * @param   [in]    dy
       * @param   [in]    dz the decomposition of the problem on each dimension. Should have np == dx*dy*dz.
       * @param   [in]    d2x
       * @param   [in]    d2y
       * @param   [in]    d2z the decomposition of the problem on each dimension. Should have d2x*d2y*d2z % np == 0.
       * @param   [in]    w The w value.
       * @param   [in]    parlog The Parallel log data structure, if parallel_log._comm == NULL, will use the global one.
       * @return     Return error message.
       */
      int            HelmholtzWithPartition(int n, int dx, int dy, int dz, int d2x, int d2y, int d2z, T w, parallel_log &parlog);
      
      /**
       * @brief   Read matrix on the host memory, matrix market format.
       * @details Read matrix on the host memory, matrix market format.
       * @note    The current version doesn't support long int as input.
       * @param   [in]    matfile The file.
       * @param   [in]    idxin The index base of the input file. 0-based or 1-based.
       * @param   [in]    parlog The parallel_log data structure.
       * @return     Return error message.
       */
      int            ReadFromSingleMMFile(const char *matfile, int idxin, parallel_log &parlog);
      
      /**
       * @brief   Plot the pattern of the parallel csr matrix using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @details Plot the pattern of the parallel csr matrix using gnuplot. Similar to spy in the MATLAB. Function for testing purpose.
       * @param [in]   datafilename The filename of the temp file holding the data.
       * @return       Return error message.
       */
      int            PlotPatternGnuPlot( const char *datafilename, int pttype = 0);
      
      /**
       * @brief   Set the diagonal complex shift for some preconditioner options.
       * @details Set the diagonal complex shift for some preconditioner options.
       * @param   [in]    diagonal_shift The complex shift will be 0+shifti.
       * @return     Return error message.
       */
      template <typename T1>
      int            SetComplexShift(T1 diagonal_shift)
      {
         this->_diag_mat.SetComplexShift(diagonal_shift);
         return PARGEMSLR_SUCCESS;
      }
      
      /**
       * @brief   Get the diagonal complex shift for some preconditioner options.
       * @details Get the diagonal complex shift for some preconditioner options.
       * @param   [in]    diagonal_shift The complex shift will be 0+shifti.
       * @return     Return error message.
       */
      template <typename T1>
      int            GetComplexShift(T1 &diagonal_shift)
      {
         this->_diag_mat.GetComplexShift(diagonal_shift);
         return PARGEMSLR_SUCCESS;
      }
      
      /** 
       * @brief   Get number of subdomains on the top level if setted in advange.
       * @details Get number of subdomains on the top level if setted in advange.
       * @return  Return the number of subdomains.
       */
      int&           GetSeparatorNumSubdomains()
      {
         return this->_separator_ndom;
      }
      
      /** 
       * @brief   Number of subdomain for each node, -1 is the separator.
       * @details Number of subdomain for each node, -1 is the separator.
       * @return  Return the number of subdomain for each node, -1 is the separator.
       */
      vector_int&    GetSeparatorDomi()
      {
         return this->_separator_domi;
      }
      
   };
   
   typedef ParallelCsrMatrixClass<float>     matrix_csr_par_float;
   typedef ParallelCsrMatrixClass<double>    matrix_csr_par_double;
   typedef ParallelCsrMatrixClass<complexs>  matrix_csr_par_complexs;
   typedef ParallelCsrMatrixClass<complexd>  matrix_csr_par_complexd;
   template<> struct PargemslrIsComplex<ParallelCsrMatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsComplex<ParallelCsrMatrixClass<complexd> > : public std::true_type {};
   template<> struct PargemslrIsParallel<ParallelCsrMatrixClass<complexs> > : public std::true_type {};
   template<> struct PargemslrIsParallel<ParallelCsrMatrixClass<complexd> > : public std::true_type {};
   
}

#endif
