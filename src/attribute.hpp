#ifndef __ATTR__
#define __ATTR__


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
#include "ICMSparser.hpp"
#include "model_management.hpp"

#include <opencv2/opencv.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// asdasd
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Attribute : public DLmodel_struct
{
public:

	model_config_t model_config;


	ea_tensor_t *rgb_LAND = NULL;

	void Init();
	void build(Parser &PARSER);
	void doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *ROI_land_crop);
	void drawLandmark(ea_display_t *display, ea_roi_t *roi);
	void postProcess();

	float *attr_output_tensor;
	std::vector<float> attr_net_out;
	bool detectionFlag;
	ea_display_t *display;

	int MaleFemale;
	int mask;
	int glasses;
	int age_YMO;
	int action_SP;
};

#endif /*__landhpp__*/