#include <stdio.h>
#include <sgx_urts.h>
#include "enclave_u.h"
#include <inttypes.h>
#include <immintrin.h>

static sgx_enclave_id_t enclave_id;

static int load_save_launch_token (const char *path, sgx_launch_token_t *launch_token, int is_save)
{
	FILE *fp = fopen(path, is_save ? "wb" : "rb");
	if (fp == NULL)
		return 1;
	size_t size;
	if (is_save)
		size = fwrite(launch_token, sizeof(sgx_launch_token_t), 1, fp);
	else
		size = fread(launch_token, sizeof(sgx_launch_token_t), 1, fp);
	fclose(fp);
	return size != sizeof(launch_token);
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		return 1;
	}
	
	uint64_t wait = atoll(argv[1]);

	// Initialize enclave
	sgx_status_t retval_init;
	sgx_launch_token_t launch_token;
	int launch_token_updated;
	retval_init = sgx_create_enclave("enclave.signed.so", SGX_DEBUG_FLAG, &launch_token, &launch_token_updated, &enclave_id, NULL);

	if (retval_init != SGX_SUCCESS) {
		printf("SGX error code: 0x%x\n", retval_init);
		return 1;
	}

	if (launch_token_updated) {
		load_save_launch_token("enclave.token", &launch_token, 1);
	}
	
	// call gadget
	uint64_t result;
	int retval;
	retval = gadget(enclave_id, &result, wait);
	if (retval != SGX_SUCCESS) {
		printf("SGX error code: 0x%x\n", retval);
	}

	printf("%"PRIu64",%"PRIu64"\n", wait, result);
}
