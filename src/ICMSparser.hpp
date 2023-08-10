#ifndef __parser_class_hpp__
#define __parser_class_hpp__

#include <eazyai.h>
#include <iostream>
#include "iniparser.hpp"



class Parser{
public:

	bool sys_visualize;
	int sys_camera_w;
	int sys_camera_h;
	int sys_log_level;


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
	int FACEID_log_level;

	std::string GAZE_ENGINE;
	float GAZE_CONFIDENCE;
	int GAZE_log_level;


	void parsing();

};

#endif	/* __parser_hpp__ */