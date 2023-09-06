#ifndef __landmarkhpp__
#define __landmarkhpp__


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
// #include "nn_arm.h"
#include "ICMSparser.hpp"
#include <opencv2/opencv.hpp>
#include "model_management.hpp"


#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <numeric>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Landmark : public DLmodel_struct
{

public:
	model_config_t model_config;


	ea_tensor_t *rgb_LAND = NULL;

	float camera_w;
	float camera_h;

	float *landmark_output_tensor;
	int landmark_img[136];
	int put_i;
	bool detectionFlag;


	struct HeadPose {
		float pitch;
		float yaw;
		float roll;
	};
	HeadPose headpose;
	HeadPose headpose_history;
	HeadPose headpose_cal;
	HeadPose headpose_cal_value;
	std::deque<HeadPose> history;
	std::vector<cv::Point2d> nose_end_point2D;
    std::deque<std::pair<float, float>> nose_history;
	float nose_result_x;
	float nose_result_y;

	int history_maxSize = 7;


	int icc_detect_is_face_valid;


	void build(Parser &PARSER);
	void doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t ROI_face_crop,int Frame_num);
	void drawLandmark(ea_display_t *display, ea_roi_t *roi);
	void postProcess(ea_roi_t *ROI_face_crop,int Frame_num);
	void headPose();
	// void headPose_asd();
	void headPoseGetAverage();
	void getAverage();


	struct TimeAndMovement
	{
		unsigned long time;
		int movement;
	};
	std::vector<TimeAndMovement> timeAndMovements;
	int ICC_DECISION_DISTRACTION;
	int icc_detect_distraction_level;
	int DISTRACTION_landmark_keep[6];
	int sumOfMovements;
	void DISTRACTION();

    int eyeOC_history_maxSize;
    std::deque<std::pair<float, float>> eyeOC_history;
	float icc_detect_left_eye_open_mm;
 	float icc_detect_right_eye_open_mm;
	int ICC_DECISION_DROWSINESS;

	int icc_detect_head_gesture;

	void calculateEARandMM(ea_roi_t *ROI_face_crop);

	int icc_detect_is_yawning;
	std::vector< unsigned long> yawning_count_vector;
	int icc_detect_driver_talking_type;
	std::vector< unsigned long> talking_count_vector;
	bool is_mouth_open;
	bool mouth_open_this_frame;
	void YAWNandSPEAKING(ea_roi_t *ROI_face_crop, int Frame_num);

	std::vector<std::pair<float, unsigned long>> All_shaking;
	int yaw_limit = 30;
	int ICC_DECISION_HEAD_GESTURE_SHAKE;

	std::vector<std::pair<float, unsigned long>> All_bob;
	int bob_limit = 1;
	int ICC_DECISION_HEAD_GESTURE_BOB;

	std::vector<std::pair<float, unsigned long>> All_nod;
	int nod_limit = 5;
	int ICC_DECISION_HEAD_GESTURE_NOD;
	void NodBobShaking();


	std::vector<TimeAndMovement> timeAnddDrow;
	int sumOfDrow;
	int icc_detect_drowsiness_level;
	void drowsiness_level(int drowsiness_val);

	void Init();

};

#endif /*__landhpp__*/


	// model_points.push_back(cv::Point3d(0.000000f, 0.000000f, 6.763430f));	//33 
	// model_points.push_back(cv::Point3d(6.825897f, 6.760612f, 4.402142f));	//17
	// model_points.push_back(cv::Point3d(1.330353f, 7.122144f, 7.122144f));	//21
	// model_points.push_back(cv::Point3d(-1.330353f, 7.122144f, 6.903745f));	//22
	// model_points.push_back(cv::Point3d(-6.825897f, 6.760612f, 4.402142f));	//26
	// model_points.push_back(cv::Point3d(5.311432f, 5.485328f, 3.987654f));	//36
	// model_points.push_back(cv::Point3d(1.789930f, 5.393625f, 4.413414f));	//39
	// model_points.push_back(cv::Point3d(-1.789930f, 5.393625f, 4.413414f));	//42
	// model_points.push_back(cv::Point3d(-5.311432f, 5.485328f, 3.987654f));	//45
	// model_points.push_back(cv::Point3d(2.005628f, 1.409845f, 6.165652f));	//31
	// model_points.push_back(cv::Point3d(-2.005628f, 1.409845f, 6.165652f));	//35
	// model_points.push_back(cv::Point3d(2.774015f, -2.080775f, 5.048531f));	//48
	// model_points.push_back(cv::Point3d(-2.774015f, -2.080775f, 5.048531f));	//54
	// model_points.push_back(cv::Point3d(0.000000f, -3.116408f, 6.097667f));	//57
	// model_points.push_back(cv::Point3d(0.000000f, -7.415691f, 4.070434f));	//8