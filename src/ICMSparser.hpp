#ifndef __parser_class_hpp__
#define __parser_class_hpp__

#include <eazyai.h>
#include <iostream>




class Parser{
public:

	bool sys_visualize;
	int sys_log_level;
	int icc_input_camera_x ; 
	int icc_input_camera_z ; 
	int icc_input_camera_y ; 
	int icc_input_camera_roll ; 
	int icc_input_camera_pitch ; 
	int icc_input_camera_yaw ; 




	std::string YOLO_ENGINE;
	float YOLO_NMS;
	float YOLO_CONFIDENCE;
	int YOLO_log_level;
	int YOLO_WIDTH;

	std::string LANDMARK_ENGINE;
	int LANDMARK_log_level;

	std::string ATTR_ENGINE;
	int ATTR_log_level;


	std::string FACEID_ENGINE;
	int FACEID_MODE_KC;
	int FACEID_MODE;
	int FACEID_log_level;

	std::string GAZE_ENGINE;
	float GAZE_CONFIDENCE;
	int GAZE_log_level;


	int camera_w;
	int camera_h;

	void parsing();

};

#endif	/* __parser_hpp__ */