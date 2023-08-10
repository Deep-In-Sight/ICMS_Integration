#include "DinsightICC.hpp"

std::queue<L1CallbackStruct> workQueue;  // Store pairs of values for A1 and B1
std::condition_variable cond_var;
std::mutex cv_m;

extern void result_value(L1CallbackStruct l1CB);  // Sum function declared in main.cpp
void ProcessQueue();




void DinsightICC::ambaSysInit(){

	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   amba system init   @@@@@@@@@@@@@@@\n");
	iav_input_param_type_t *iav_input = NULL;
	int features = 0;
	ea_display_params_t display_params = {0};
	features = EA_ENV_ENABLE_CAVALRY | EA_ENV_ENABLE_VPROC | EA_ENV_ENABLE_NNCTRL | EA_ENV_ENABLE_IAV;
	features |= EA_ENV_ENABLE_OSD_VOUT; // if vout?? else stream
	ea_env_open(features); // Open the CV environment with features

	iav_input = (iav_input_param_type_t *)malloc(sizeof(iav_input_param_type_t));
	memset(iav_input, 0, sizeof(iav_input_param_type_t));
	iav_input->img_resource[0] = ea_img_resource_new(EA_CANVAS, (void *)1);	 // use pyramid?? //set canvas & canvas_id// return : A pointer which represents a image resource.
	iav_input->id = 0;
	iav_input->type = EA_COLOR_YUV2RGB_NV12; // color RGB_PLANAR?
	iav_input->crop_type = EA_TENSOR_COLOR_MODE_RGB;
	input_param = iav_input;
	display = ea_display_new(EA_DISPLAY_VOUT, 1, 1, NULL); // OUT_TYPE_VOUT?? //set display env
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n");

	memset(&calc_fps_ctx, 0, sizeof(ea_calc_fps_ctx_t));
	calc_fps_ctx.count_period = DEFAULT_FPS_COUNT_PERIOD;

}




ea_display_t* DinsightICC::sysInit()
{
	EA_LOG_SET_LOCAL(ParserObj.sys_log_level);

	EA_LOG_NOTICE("MAIN Start ICC Init\n");
	ambaSysInit();
	EA_LOG_NOTICE("MAIN Start FACE.build\n");
	FACE.build(ParserObj);
	EA_LOG_NOTICE("MAIN Start LANDMARK.build\n");
	LANDMARK.build(ParserObj);
	EA_LOG_NOTICE("MAIN Start ATTR.build\n");
	ATTR.build(ParserObj);
	EA_LOG_NOTICE("MAIN Start FACEID.build\n");
	FACEID.build(ParserObj);
	return display;
}

void DinsightICC::DModelInit(){
	ATTR.Init();

}







void DinsightICC::StartICC(){
	std::thread t(ProcessQueue);
	do
	{
		DModelInit();


		ea_img_resource_hold_data(input_param->img_resource[0], &input_param->data); // canvas & canvas_id로 설정된 구조체의 포인트, 저장될 곳  (구조체 : ea_img_resource_s)

		FACE.doInference(&input_param->data.tensor_group[0],&ROI_land_crop );
		// FACE.draw_detection_bbox_textbox(display);
		if (FACE.res->valid_det_count>0)
		{
			LANDMARK.doInference(&input_param->data.tensor_group[0], &ROI_land_crop);
			// LANDMARK.drawLandmark(display, &ROI_land_crop);

			ATTR.doInference(&input_param->data.tensor_group[0], &ROI_land_crop);
			// ATTR.drawLandmark(display, &ROI_land_crop);

			FACEID.doInference(&input_param->data.tensor_group[0], &ROI_land_crop);
			// FACEID.drawLandmark(display, &ROI_land_crop);
		}

		ea_img_resource_drop_data(input_param->img_resource[0], &(input_param->data));//hold data max 데이터의 수가 최대 50개//이미지 데이터가 내부 버퍼에서 해제
		ea_display_refresh(display, 0); // 그린거 다시 비우기, 이거 안하면 이전 그린거 유지됨
		fps = ea_calc_fps(&calc_fps_ctx);


		if (fps > 0)
		{
			EA_LOG_DEBUG("fps %.1f\n", fps);
		}

		resultSetStruct();

		std::lock_guard<std::mutex> lk(cv_m);
        workQueue.push({l1CB});
        cond_var.notify_all();



	} while (1);


}

void DinsightICC::resultSetStruct(){

	l1CB.head_rect.x_start= FACE.tracked_face.x_start;
	l1CB.head_rect.y_start= FACE.tracked_face.y_start;
	l1CB.head_rect.x_end= FACE.tracked_face.x_end;
	l1CB.head_rect.y_end= FACE.tracked_face.y_end;
	l1CB.is_driver_valid=FACE.is_driver_valid;

	l1CB.ATTR_zip.MaleFemale=ATTR.MaleFemale;
	l1CB.ATTR_zip.mask=ATTR.mask;
	l1CB.ATTR_zip.glasses=ATTR.glasses;
	l1CB.ATTR_zip.age_YMO=ATTR.age_YMO;
	l1CB.ATTR_zip.action_SP=ATTR.action_SP;

	l1CB.landmark_img = LANDMARK.landmark_img;


}


void ProcessQueue() {
    while (true) {
        std::unique_lock<std::mutex> lk(cv_m);
        cond_var.wait(lk, []{return !workQueue.empty();});
        L1CallbackStruct item = workQueue.front();
        workQueue.pop();
        lk.unlock();
        result_value(item);
    }
}


