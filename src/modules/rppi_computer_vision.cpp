#include <rppi_computer_vision.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>
#include "hip/hip_declarations.hpp"

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_computer_vision.hpp" 

RppStatus  
rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		data_object_copy_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		data_object_copy_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		data_object_copy_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		data_object_copy_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		data_object_copy_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	data_object_copy_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	data_object_copy_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	data_object_copy_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_data_object_copy_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	data_object_copy_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_ROI_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		local_binary_pattern_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		local_binary_pattern_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	local_binary_pattern_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	local_binary_pattern_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,rppHandle_t rppHandle )
{ 
	local_binary_pattern_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_ROI_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_roi(roiPoints, rpp::deref(rppHandle));
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_local_binary_pattern_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	local_binary_pattern_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		roiPoints,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}



// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		gaussian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		gaussian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	gaussian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	gaussian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	gaussian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	gaussian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 1
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PLANAR, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl(
// 			static_cast<cl_mem>(srcPtr),
// 			 srcSize,
// 			static_cast<cl_mem>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			,rpp::deref(rppHandle));
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			,rpp::deref(rppHandle));
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
// 	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);

// #ifdef OCL_COMPILE
// 	{
// 		laplacian_image_pyramid_cl_batch(
// 			static_cast<cl_mem>(srcPtr),
// 			static_cast<cl_mem>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #elif defined (HIP_COMPILE)
// 	{
// 		laplacian_image_pyramid_hip_batch(
// 			static_cast<Rpp8u*>(srcPtr),
// 			static_cast<Rpp8u*>(dstPtr),
// 			rpp::deref(rppHandle),
// 			RPPI_CHN_PACKED, 3
// 		);
// 	}
// #endif //BACKEND

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	laplacian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	laplacian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,rppHandle_t rppHandle )
// { 
// 	laplacian_image_pyramid_host(
// 			static_cast<Rpp8u *>(srcPtr),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			stdDev,
// 			kernelSize,
// 			RPPI_CHN_PACKED, 3 
// 			);


// 	return RPP_SUCCESS;
// }
// template <typename T>
// RppStatus canny_edge_detector_host_batch(T* batch_srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* batch_dstPtr, 
//                                          T *batch_maxThreshold, T *batch_minThreshold, 
//                                          Rpp32u nbatchSize, 
//                                          RppiChnFormat chnFormat, Rpp32u channel)
// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp8u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
// 	copy_param_uchar (kernelSize, rpp::deref(rppHandle), paramIndex++);
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp8u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	laplacian_image_pyramid_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
// 		stdDev,
// 		kernelSize,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

RppStatus  
rppi_canny_edge_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		canny_edge_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		canny_edge_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 
	canny_edge_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 
	canny_edge_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,rppHandle_t rppHandle )
{ 
	canny_edge_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			minThreshold,
			maxThreshold,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u minThreshold ,Rpp8u maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uchar (minThreshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (maxThreshold, rpp::deref(rppHandle), paramIndex++);
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[0].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_canny_edge_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp8u *minThreshold ,Rpp8u *maxThreshold ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	canny_edge_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		minThreshold,
		maxThreshold,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		harris_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		harris_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	harris_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	harris_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	harris_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			gaussianKernelSize,
			stdDev,
			kernelSize,
			kValue,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u gaussianKernelSize ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32f kValue ,Rpp32f threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (gaussianKernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (kValue, rpp::deref(rppHandle), paramIndex++);
	copy_param_float (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[1].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[3].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[4].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[5].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_harris_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *gaussianKernelSize ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32f *kValue ,Rpp32f *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	harris_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		gaussianKernelSize,
		stdDev,
		kernelSize,
		kValue,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		fast_corner_detector_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		fast_corner_detector_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	fast_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	fast_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,rppHandle_t rppHandle )
{ 
	fast_corner_detector_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			numOfPixels,
			threshold,
			nonmaxKernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u numOfPixels ,Rpp8u threshold ,Rpp32u nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_uint (numOfPixels, rpp::deref(rppHandle), paramIndex++);
	copy_param_uchar (threshold, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (nonmaxKernelSize, rpp::deref(rppHandle), paramIndex++);
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[0].uintmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.ucharArr[1].ucharmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[2].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_fast_corner_detector_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u *numOfPixels ,Rpp8u *threshold ,Rpp32u *nonmaxKernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	fast_corner_detector_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		numOfPixels,
		threshold,
		nonmaxKernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

/*RppStatus  
rppi_control_flow_u8_pln1_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		control_flow_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			type,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			type,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
            type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
            type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		control_flow_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			type,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			type,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		control_flow_cl(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			type,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			type,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_ROI_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	RppiROI roiPoints;
	roiPoints.x = 0;
	roiPoints.y = 0;
	roiPoints.roiHeight = 0;
	roiPoints.roiWidth = 0;
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPS_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPD_ROIS_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPS_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchSD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchDD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pkd3_batchPD_ROID_gpu(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	copy_roi(roiPoints, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		control_flow_cl_batch(
			static_cast<cl_mem>(srcPtr1),
			static_cast<cl_mem>(srcPtr2),
			static_cast<cl_mem>(dstPtr),
type,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		control_flow_hip_batch(
			static_cast<Rpp8u*>(srcPtr1),
			static_cast<Rpp8u*>(srcPtr2),
			static_cast<Rpp8u*>(dstPtr),
            type,			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_control_flow_u8_pln1_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
{ 
	control_flow_host(
			static_cast<Rpp8u *>(srcPtr1),
			static_cast<Rpp8u *>(srcPtr2),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			type,
			RPPI_CHN_PLANAR, 1 
			);
	return RPP_SUCCESS;
}

// RppStatus  
// rppi_control_flow_u8_pln1_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,	
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln1_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 1
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
// { 
// 	control_flow_host(
// 			static_cast<Rpp8u *>(srcPtr1),
// 			static_cast<Rpp8u *>(srcPtr2),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			type,
// 			RPPI_CHN_PLANAR, 1 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pln3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PLANAR, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,rppHandle_t rppHandle )
// { 
// 	control_flow_host(
// 			static_cast<Rpp8u *>(srcPtr1),
// 			static_cast<Rpp8u *>(srcPtr2),
// 			 srcSize,
// 			static_cast<Rpp8u *>(dstPtr),
// 			type,
// 			RPPI_CHN_PACKED, 3 
// 			);


// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_ROI_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchPS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchPD_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	RppiROI roiPoints;
// 	roiPoints.x = 0;
// 	roiPoints.y = 0;
// 	roiPoints.roiHeight = 0;
// 	roiPoints.roiWidth = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchPS_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchPD_ROIS_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_roi(roiPoints, rpp::deref(rppHandle));
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//         type,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }


// RppStatus  
// rppi_control_flow_u8_pkd3_batchPS_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	copy_param_uint (type, rpp::deref(rppHandle), paramIndex++);
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchSD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchDD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		srcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             type,		
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

// 	return RPP_SUCCESS;
// }

// RppStatus  
// rppi_control_flow_u8_pkd3_batchPD_ROID_host(RppPtr_t srcPtr1 ,RppPtr_t srcPtr2 ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u type ,RppiROI *roiPoints ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
// { 
// 	Rpp32u paramIndex = 0;
// 	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
// 	control_flow_host_batch<Rpp8u>(
// 		static_cast<Rpp8u*>(srcPtr1),
// 		static_cast<Rpp8u*>(srcPtr2),
// 		srcSize,
// 		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
// 		static_cast<Rpp8u*>(dstPtr),
//             		type,
// 		roiPoints,
// 		rpp::deref(rppHandle).GetBatchSize(),
// 		RPPI_CHN_PACKED, 3
// 	);

	return RPP_SUCCESS;
}
*/

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		gaussian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		gaussian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	gaussian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_gaussian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	gaussian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);

#ifdef OCL_COMPILE
	{
		laplacian_image_pyramid_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		laplacian_image_pyramid_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	laplacian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	laplacian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,rppHandle_t rppHandle )
{ 
	laplacian_image_pyramid_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			stdDev,
			kernelSize,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f stdDev ,Rpp32u kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	copy_param_float (stdDev, rpp::deref(rppHandle), paramIndex++);
	copy_param_uint (kernelSize, rpp::deref(rppHandle), paramIndex++);
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.floatArr[0].floatmem,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.uintArr[1].uintmem,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_laplacian_image_pyramid_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32f *stdDev ,Rpp32u *kernelSize ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	laplacian_image_pyramid_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr),
		stdDev,
		kernelSize,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}


RppStatus  
rppi_remap_u8_pln1_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		remap_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			 rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			 rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr),
			 rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr),
			 rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 1
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		remap_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			 rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			  rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PLANAR);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PLANAR, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 

#ifdef OCL_COMPILE
	{
		remap_cl(
			static_cast<cl_mem>(srcPtr),
			 srcSize,
			static_cast<cl_mem>(dstPtr),
			  rowRemapTable, colRemapTable,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 rowRemapTable, colRemapTable,
			RPPI_CHN_PACKED, 3 
			,rpp::deref(rppHandle));
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchSS_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchSD_gpu(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchDS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchDD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPS_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_srcSize(srcSize, rpp::deref(rppHandle));
	copy_srcMaxSize (maxSrcSize, rpp::deref(rppHandle));
	get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

#ifdef OCL_COMPILE
	{
		remap_cl_batch(
			static_cast<cl_mem>(srcPtr),
			static_cast<cl_mem>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#elif defined (HIP_COMPILE)
	{
		remap_hip_batch(
			static_cast<Rpp8u*>(srcPtr),
			static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
			rpp::deref(rppHandle),
			RPPI_CHN_PACKED, 3
		);
	}
#endif //BACKEND

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 
	remap_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			  rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln1_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 1
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 
	remap_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			rowRemapTable, colRemapTable,
			RPPI_CHN_PLANAR, 1 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pln3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PLANAR, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,rppHandle_t rppHandle )
{ 
	remap_host(
			static_cast<Rpp8u *>(srcPtr),
			 srcSize,
			static_cast<Rpp8u *>(dstPtr),
			 rowRemapTable, colRemapTable,
			RPPI_CHN_PACKED, 3 
			);


	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchSS_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchSD_host(RppPtr_t srcPtr ,RppiSize srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_srcSize(srcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchDS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchDD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		srcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPS_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}

RppStatus  
rppi_remap_u8_pkd3_batchPD_host(RppPtr_t srcPtr ,RppiSize *srcSize ,RppiSize maxSrcSize ,RppPtr_t dstPtr ,Rpp32u * rowRemapTable ,Rpp32u * colRemapTable ,Rpp32u nbatchSize ,rppHandle_t rppHandle )
{ 
	Rpp32u paramIndex = 0;
	copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));
	remap_host_batch<Rpp8u>(
		static_cast<Rpp8u*>(srcPtr),
		srcSize,
		rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
		static_cast<Rpp8u*>(dstPtr), rowRemapTable, colRemapTable,
		rpp::deref(rppHandle).GetBatchSize(),
		RPPI_CHN_PACKED, 3
	);

	return RPP_SUCCESS;
}
