#include "DinsightICC.hpp"

#include "face.hpp"
#include "landmark.hpp"
#include "gaze.hpp"

extern std::string save_num;
extern std::string save_img;


std::queue<ea_tensor_t*> faceQueue;
std::queue<ea_tensor_t*> IimgQueue;
std::mutex faceMutex;
std::mutex IimgMutex;


void DinsightICC::initClassFunction()
{
	FACE = new Face();
	LANDMARK = new Landmark();
	GAZE = new Gaze();


}

void DinsightICC::ambaSysInit()
{
	initClassFunction();
	Frame_num = 0;
	int features = 0;
	features = EA_ENV_ENABLE_CAVALRY | EA_ENV_ENABLE_VPROC | EA_ENV_ENABLE_NNCTRL | EA_ENV_ENABLE_IAV;
	features |= EA_ENV_ENABLE_OSD_VOUT; // if vout?? else stream
	ea_env_open(features);				// Open the CV environment with features

	display = ea_display_new(EA_DISPLAY_VOUT, 1, 1, NULL); // OUT_TYPE_VOUT?? //set display env
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n");

	memset(&calc_fps_ctx, 0, sizeof(ea_calc_fps_ctx_t));
	calc_fps_ctx.count_period = DEFAULT_FPS_COUNT_PERIOD;
}

void DinsightICC::sysInit()
{
    startICCTread = std::thread(&DinsightICC::StartICC, this);
    // t = std::thread(ProcessQueue); // 초기화 함수에서 스레드 생성

	EA_LOG_SET_LOCAL(ParserObj.sys_log_level);
	EA_LOG_NOTICE("MAIN Start ICC Init\n");
	ambaSysInit();
	EA_LOG_NOTICE("MAIN Start FACE.build\n");
	FACE->build(ParserObj);	
	EA_LOG_NOTICE("MAIN Start LANDMARK.build\n");
	LANDMARK->build(ParserObj);
	EA_LOG_NOTICE("MAIN Start GAZE.build\n");
	GAZE->build(ParserObj); //BLINKEYE1


	size_t face_s1[] = {1, 1, 320, 320*3};
	face_Iimg_tensor1 = ea_tensor_new(EA_U8, face_s1, 0);
		faceCV = cv::Mat(320, 320, CV_8UC3, ea_tensor_data(face_Iimg_tensor1), ea_tensor_pitch(face_Iimg_tensor1));
	size_t face_s2[] = {1, 3, 320, 320};
	face_Iimg_tensor2 = ea_tensor_new(EA_U8, face_s2, 0);

	size_t Iimg_s1[] = {1, 1, 800, 800*3};
	Iimg_tensor1 = ea_tensor_new(EA_U8, Iimg_s1, 0);
		Iimg = cv::Mat(800, 800, CV_8UC3, ea_tensor_data(Iimg_tensor1), ea_tensor_pitch(Iimg_tensor1));
	size_t Iimg_s2[] = {1, 3, 800, 800};
	Iimg_tensor2 = ea_tensor_new(EA_U8, Iimg_s2, 0);

}

void DinsightICC::DModelInit()
{
	LANDMARK->Init();
	FACE->Init();
	// GAZE->Init();

	ROI_face_crop = {-1, -1, -1, -1};
}



const int MAX_QUEUE_SIZE = 3;

void DinsightICC::imgQ(cv::Mat cam_img)
{


	cv::Mat first_source;
	cv::Rect crop_region(240, 0, 800, 800);
    first_source = cam_img(crop_region);
    cv::Mat face_Iimg_cv;
    cv::Size size(320, 320);
    cv::resize(first_source, face_Iimg_cv, size);
    cv::cvtColor(face_Iimg_cv, face_Iimg_cv, cv::COLOR_GRAY2BGR);
	
	
	face_Iimg_cv.copyTo(faceCV);
	ea_cvt_color_resize(face_Iimg_tensor1, face_Iimg_tensor2, EA_COLOR_TRANSPOSE, EA_VP);
    cv::Mat Iimg_cv;
    cv::cvtColor(first_source, Iimg_cv, cv::COLOR_GRAY2BGR);
	Iimg_cv.copyTo(Iimg);
	ea_cvt_color_resize(Iimg_tensor1, Iimg_tensor2, EA_COLOR_TRANSPOSE, EA_VP);

	//img q push
	std::lock_guard<std::mutex> faceGuard(faceMutex);
	if (faceQueue.size() >= MAX_QUEUE_SIZE)
	{
		std::cout << "img pop pop" << std::endl;
		faceQueue.pop();
	}
	faceQueue.push(face_Iimg_tensor2);
	std::lock_guard<std::mutex> IimgGuard(IimgMutex);
	if (IimgQueue.size() >= MAX_QUEUE_SIZE)
	{
		std::cout << "img pop pop" << std::endl;
		IimgQueue.pop();
	}
	IimgQueue.push(Iimg_tensor2);
}


void DinsightICC::StartICC()
{

	while (true) // 또는 종료 조건이 충족될 때까지
	{
		ea_tensor_t *currentFaceTensor = nullptr;
		ea_tensor_t *currentIimgTensor = nullptr;

		std::lock_guard<std::mutex> faceGuard(faceMutex);
		if (!faceQueue.empty())
		{
			currentFaceTensor = faceQueue.front();
			faceQueue.pop();
		}
		std::lock_guard<std::mutex> IimgGuard(IimgMutex);
		if (!IimgQueue.empty())
		{
			currentIimgTensor = IimgQueue.front();
			IimgQueue.pop();
		}

		if (currentFaceTensor && currentIimgTensor)
		{

			DModelInit();
			Frame_num++;

			FACE->doInference(&face_Iimg_tensor2, &ROI_face_crop);
			if (FACE->icc_detect_is_face_valid == true)
			{
				LANDMARK->doInference(&Iimg_tensor2, ROI_face_crop, Frame_num);
				GAZE->doInference(&Iimg_tensor2, ROI_face_crop,LANDMARK->landmark_img); //BLINKEYE2
			}

			//BLINKEYE3
			//visualization & img save
			std::cout << "Frame_num				" << Frame_num << std::endl;
			std::cout << "FACE->icc_detect_is_face_valid		" << FACE->icc_detect_is_face_valid << std::endl;
			std::cout << "ROI_face_crop.x				" << ROI_face_crop.x << std::endl;
			std::cout << "ROI_face_crop.y				" << ROI_face_crop.y << std::endl;
			std::cout << "ROI_face_crop.x + ROI_face_crop.w	" << ROI_face_crop.x + ROI_face_crop.w << std::endl;
			std::cout << "ROI_face_crop.y + ROI_face_crop.h	" << ROI_face_crop.y + ROI_face_crop.h << std::endl;
			std::cout << "GAZE->icc_detect_gaze.pitch		" << GAZE->icc_detect_gaze.pitch << std::endl;
			std::cout << "GAZE->icc_detect_gaze.yaw		" << GAZE->icc_detect_gaze.yaw << std::endl;
			ea_tensor_sync_cache(Iimg_tensor2, EA_VP, EA_CPU);
			cv::Mat to_opencv_img(800, 800, CV_8UC1, ea_tensor_data(Iimg_tensor2), ea_tensor_pitch(Iimg_tensor2));
			cv::Mat deepIMG = to_opencv_img.clone();
			cv::cvtColor(deepIMG, deepIMG, cv::COLOR_GRAY2BGR);
			if (FACE->icc_detect_is_face_valid == true)
			{
				cv::rectangle(deepIMG, cv::Rect(ROI_face_crop.x - 240, ROI_face_crop.y, ROI_face_crop.w, ROI_face_crop.h), cv::Scalar(0, 0, 255), 5, 8, 0);
				for (int i = 0; i < 68; ++i)
				{
					int x = LANDMARK->landmark_img[i * 2];
					int y = LANDMARK->landmark_img[i * 2 + 1];
					cv::circle(deepIMG, cv::Point(x, y), 4, cv::Scalar(0, 0, 255), -1);
				}
			}
			std::string filename = save_img + save_num+ std::to_string(Frame_num) + ".jpg";
					std::cout<<"filename  "<<filename<<std::endl;
			cv::imwrite(filename, deepIMG);
		}
	}
}
