#include "enclave_u.h"
#include <errno.h>

typedef struct ms_gadget_t {
	uint64_t ms_retval;
	uint64_t ms_received_val_for_train;
} ms_gadget_t;

static const struct {
	size_t nr_ocall;
	void * table[1];
} ocall_table_enclave = {
	0,
	{ NULL },
};
sgx_status_t gadget(sgx_enclave_id_t eid, uint64_t* retval, uint64_t received_val_for_train)
{
	sgx_status_t status;
	ms_gadget_t ms;
	ms.ms_received_val_for_train = received_val_for_train;
	status = sgx_ecall(eid, 0, &ocall_table_enclave, &ms);
	if (status == SGX_SUCCESS && retval) *retval = ms.ms_retval;
	return status;
}

