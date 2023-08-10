#ifndef __faceID__
#define __faceID__


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
#include "model_management.hpp"
#include "ICMSparser.hpp"

#include <opencv2/opencv.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FaceID : public DLmodel_struct
{
public:

	model_config_t model_config;


	ea_tensor_t *rgb_LAND = NULL;
	ea_tensor_t *grey_LAND = NULL;

	void Init();
	void build(Parser &PARSER);
	void doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *ROI_land_crop);
	void drawLandmark(ea_display_t *display, ea_roi_t *roi);

	float *faceid_output_tensor;

	bool detectionFlag;
};

#endif /*__landhpp__*/