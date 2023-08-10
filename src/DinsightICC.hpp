#ifndef __DinsightICChpp__
#define __DinsightICChpp__

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

#include <opencv2/opencv.hpp>
#include "ICMSparser.hpp"

#include <future>
#include <cstdlib>

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_VERBOSE);
EA_MEASURE_TIME_DECLARE();
#include "model_management.hpp"
#include "face.hpp"
#include "landmark.hpp"
#include "attribute.hpp"
#include "faceID.hpp"
#include "DinsightCallbackStruct.hpp"





#define DEFAULT_FPS_COUNT_PERIOD 100   // show fps per factor

// cvflow
#define NN_MAX_PORT_NUM 20
#define NN_MAX_STR_LEN 256

#define MAX_PATH_STRLEN 256
#define MAX_LABEL_NUM 256



class DinsightICC : public Parser
{
public:

	typedef struct iav_input_param_type_s
	{
		ea_img_resource_t *img_resource[NN_MAX_PORT_NUM];
		ea_tensor_t *y_tensor;
		ea_color_convert_type_t type;
		ea_img_resource_data_t data;
		int id;
		int y_id;
		int uv_id;
		int yuv_require;
		// hold_data_fun_t hold_data;
		ea_tensor_color_mode_t crop_type;
	} iav_input_param_type_t;

	ea_calc_fps_ctx_t calc_fps_ctx;
	float fps;


    Parser& ParserObj;
    DinsightICC(Parser& Parser_ref) : ParserObj(Parser_ref) {}


	iav_input_param_type_t *input_param;
	ea_display_t *display;
	L1CallbackStruct l1CB;
	
	Face FACE;
	Landmark LANDMARK;
	Attribute ATTR;
	FaceID FACEID;


	ea_roi_t ROI_land_crop;



	ea_display_t* sysInit();
	void ambaSysInit();
	void StartICC();
	void resultSetStruct();
	void DModelInit();

};



#endif /*__DinsightICC__*/
