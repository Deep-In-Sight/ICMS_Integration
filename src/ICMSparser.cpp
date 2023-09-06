#include <iostream>
#include "ICMSparser.hpp"
#include "iniparser.hpp"

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);


void Parser::parsing()
{

	std::string fullpath = "./config.ini";
	INI::File ft;
	ft.Load(fullpath);

	sys_visualize = 	ft.GetSection("SYS")->GetValue("visualize", -1).AsBool();
	sys_log_level = 	ft.GetSection("SYS")->GetValue("log_level", -1).AsInt();
	icc_input_camera_x = 		ft.GetSection("SYS")->GetValue("icc_input_camera_x",-1).AsInt() ; 
	icc_input_camera_z = 		ft.GetSection("SYS")->GetValue("icc_input_camera_z",-1).AsInt() ; 
	icc_input_camera_y = 		ft.GetSection("SYS")->GetValue("icc_input_camera_y",-1).AsInt() ; 
	icc_input_camera_roll = 	ft.GetSection("SYS")->GetValue("icc_input_camera_roll",-1).AsInt() ; 
	icc_input_camera_pitch =	ft.GetSection("SYS")->GetValue("icc_input_camera_pitch",-1).AsInt() ; 
	icc_input_camera_yaw = 		ft.GetSection("SYS")->GetValue("icc_input_camera_yaw",-1).AsInt() ; 

	YOLO_ENGINE = 		ft.GetSection("YOLO")->GetValue("ENGINE", -1).AsString();
	YOLO_ENGINE = 		"./" + YOLO_ENGINE + ".bin";
	YOLO_NMS = 			ft.GetSection("YOLO")->GetValue("NMS", -1).AsFloat();
	YOLO_CONFIDENCE = 	ft.GetSection("YOLO")->GetValue("CONFIDENCE", -1).AsFloat();
	YOLO_WIDTH = 		ft.GetSection("YOLO")->GetValue("WIDTH", -1).AsInt();
	YOLO_log_level = 	ft.GetSection("YOLO")->GetValue("log_level", -1).AsInt();


	LANDMARK_ENGINE = 	ft.GetSection("LANDMARK")->GetValue("ENGINE", -1).AsString();
	LANDMARK_ENGINE = 	"./" + LANDMARK_ENGINE + ".bin";
	LANDMARK_log_level =ft.GetSection("LANDMARK")->GetValue("log_level", -1).AsInt();

	ATTR_ENGINE = 		ft.GetSection("ATTR")->GetValue("ENGINE", -1).AsString();
	ATTR_ENGINE = 		"./" + ATTR_ENGINE + ".bin";
	ATTR_log_level = 	ft.GetSection("ATTR")->GetValue("log_level", -1).AsInt();

	FACEID_ENGINE = 	ft.GetSection("FACEID")->GetValue("ENGINE", -1).AsString();
	FACEID_ENGINE = 	"./" + FACEID_ENGINE + ".bin";
	FACEID_MODE = 		ft.GetSection("FACEID")->GetValue("SET_MODE", -1).AsInt();
	FACEID_MODE_KC = 		ft.GetSection("FACEID")->GetValue("MODE", -1).AsInt();
	FACEID_log_level = 	ft.GetSection("FACEID")->GetValue("log_level", -1).AsInt();


// FACEID_MODE_KC
// FACEID_MODE


	GAZE_ENGINE = 		ft.GetSection("GAZE")->GetValue("ENGINE", -1).AsString();
	GAZE_ENGINE = 		"./" + GAZE_ENGINE + ".bin";
	GAZE_CONFIDENCE = 	ft.GetSection("GAZE")->GetValue("CONFIDENCE", -1).AsFloat();
	GAZE_log_level = 	ft.GetSection("GAZE")->GetValue("log_level", -1).AsInt();

	EA_LOG_SET_LOCAL(sys_log_level);

	std::cout << "  " << std::endl;
	EA_LOG_DEBUG("###### <<ICMS Parser Infor>>######\n");

	EA_LOG_DEBUG("sys_visualize : %d\n",sys_visualize);
	EA_LOG_DEBUG("sys_log_level : %d\n",sys_log_level);

	EA_LOG_DEBUG("##################################\n");

	EA_LOG_DEBUG("YOLO_ENGINE : %s\n", YOLO_ENGINE.c_str());
	EA_LOG_DEBUG("YOLO_NMS : %f\n", YOLO_NMS);
	EA_LOG_DEBUG("YOLO_CONFIDENCE : %f\n", YOLO_CONFIDENCE);
	EA_LOG_DEBUG("WIDTH : %d\n", YOLO_WIDTH);
	EA_LOG_DEBUG("YOLO_log_level : %d\n", YOLO_log_level);

	EA_LOG_DEBUG("##################################\n");

	EA_LOG_DEBUG("LANDMARK_ENGINE : %s\n",LANDMARK_ENGINE.c_str() );
	EA_LOG_DEBUG("LANDMARK_log_level : %d\n",LANDMARK_log_level );

	EA_LOG_DEBUG("##################################\n");

	EA_LOG_DEBUG("ATTR_ENGINE : %d\n",ATTR_ENGINE.c_str());
	EA_LOG_DEBUG("ATTR_log_level : %d\n",ATTR_log_level);

	EA_LOG_DEBUG("##################################\n");

	EA_LOG_DEBUG("FACEID_ENGINE : %d\n",FACEID_ENGINE.c_str());
	EA_LOG_DEBUG("FACEID_log_level : %d\n",FACEID_log_level);

	EA_LOG_DEBUG("##################################\n");
	std::cout << "  " << std::endl;

}