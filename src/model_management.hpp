#ifndef __manage__
#define __manage__


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <signal.h>
#include <getopt.h>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <eazyai.h>
#define NN_MAX_PORT_NUM 20

class DLmodel_struct
{
public:

	typedef struct nn_model_port_info_s
	{
		const char *name;
		ea_tensor_t *tensor;
	} nn_model_port_info_t;

	typedef struct out_info_s
	{
		const char *tensor_name; /*!< Name of vp output. */
		ea_tensor_t *out;		 /*!< A pointer to represent the tensor which shares the output data. */
	} out_info_t;

	typedef struct model_config_s
	{
		ea_net_t *net;
		int in_num;
		int out_num;
		nn_model_port_info_t in_info[NN_MAX_PORT_NUM];
		nn_model_port_info_t out_info[NN_MAX_PORT_NUM];

		out_info_t *out;			/*!< A pointer array to hold output. */
		uint64_t seq;				/*!< The sequence number of result. */
		unsigned long timestamp_us; /*!< The timestamp of result. */

	} model_config_t;
};
#endif /*__manage__*/