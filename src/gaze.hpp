#ifndef __GAZE__
#define __GAZE__


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

class Gaze : public DLmodel_struct //BLINKEYE
{
public:

	model_config_t model_config;



	ea_tensor_t *rgb_LAND = NULL;
	int Freq;
	int Freq_c;


	float *gaze_output_tensor;
	std::vector<float> gaze_net_out;
	ea_display_t *display;

	int icc_detect_R_eye_gaze_conf;
	int icc_detect_L_eye_gaze_conf;

	struct icc_detect_gaze_struct
	{
		float pitch;
		float yaw;
	};
	icc_detect_gaze_struct icc_detect_gaze;
	icc_detect_gaze_struct icc_detect_R_eye_gaze;
	icc_detect_gaze_struct icc_detect_L_eye_gaze;
	icc_detect_gaze_struct icc_input_camera;
	icc_detect_gaze_struct icc_detect_R_eye_gaze_cal;
	icc_detect_gaze_struct icc_detect_L_eye_gaze_cal;
	icc_detect_gaze_struct icc_detect_gaze_cal;

	int icc_detect_eyes_on_road;

	std::vector< unsigned long> eyesOnRoad_count_vector;

	ea_roi_t ROI_attr_xywh;


	void Init();
	void build(Parser &PARSER);
	void doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t ROI_face_crop,int *landmark_img);
	void drawLandmark(ea_display_t *display, ea_roi_t *roi);
	void postProcess();
};

#endif /*__GAZE__*/