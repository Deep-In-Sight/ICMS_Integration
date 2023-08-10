#ifndef __DinsightCallbackStruct__
#define __DinsightCallbackStruct__

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

struct face_result // 0~1
{
	float x_start;
	float y_start;
	float x_end;
	float y_end;
};

struct ATTR_struct
{
	int MaleFemale;
	int mask;
	int glasses;
	int age_YMO;
	int action_SP;
};

typedef struct L1CallbackStruct
{
	face_result head_rect;
	int is_driver_valid;
	ATTR_struct ATTR_zip;
	float *landmark_img;

};

#endif /*__DinsightICC__*/
