#include "DinsightICC.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

std::string save_num ="0/";
std::string save_img = "./img_source/output/";
std::string read_img = "./img_source/input/";



int main()
{


	Parser ParserObj;
	ParserObj.parsing();
	ParserObj.camera_h = 800;
	ParserObj.camera_w = 1280;

	EA_LOG_SET_LOCAL(ParserObj.sys_log_level);
	DinsightICC ICC(ParserObj);
	EA_LOG_NOTICE("MAIN Start sysInit\n");
	ICC.sysInit();
	EA_LOG_NOTICE("MAIN Start StartICC\n");

	int save_c = 0;

	while (true)
	{
		unsigned long t1 = ea_gettime_us();

		save_c++;
		std::string filename = read_img +save_num+ std::to_string(save_c) + ".jpg";
		cv::Mat frame_image_8bit = cv::imread(filename, cv::IMREAD_GRAYSCALE);

		unsigned long t2 = ea_gettime_us();

		ICC.imgQ(frame_image_8bit);


	}

	// ICC.StartICC();

	return 0;
}