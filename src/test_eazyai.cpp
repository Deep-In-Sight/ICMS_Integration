#include "DinsightICC.hpp"
ea_display_t *display;

int main()
{
    Parser ParserObj;
    ParserObj.parsing();

	EA_LOG_SET_LOCAL(ParserObj.sys_log_level);

	DinsightICC ICC(ParserObj);
	EA_LOG_NOTICE("MAIN Start sysInit\n");
	display=ICC.sysInit();
	EA_LOG_NOTICE("MAIN Start StartICC\n");
	ICC.StartICC();

	/*
	DinsightICC ICCR;
	ICCR.sysInit();
	ICCR.StartICC();
	*/

	return 0;
}






void displayHandler(int condition, const char* default_text, std::map<int, const char*> text_map, float& y, float x, float w, float h) {
	ea_display_obj_params(display)->text_color = (condition == -1) ? EA_16_COLORS_BLUE : EA_16_COLORS_RED;
	ea_display_set_textbox(display, (condition == -1) ? default_text : text_map[condition], x, y, w, h);
	y += 0.03;
}

std::map<int, const char *> gender_text_map = {{0, "Male"}, {1, "Female"}};
std::map<int, const char *> mask_text_map = {{0, "Mask_NO"}, {1, "Mask"}};
std::map<int, const char *> glasses_text_map = {{0, "Glasses_NO"}, {1, "Glasses"}};
std::map<int, const char *> age_YMO_text_map = {{0, "Young"}, {1, "Middle"}, {2, "Old"}};
std::map<int, const char *> action_SP_text_map = {{0, "Normal"}, {1, "Smoking"}, {2, "Phone"}};



void result_value(L1CallbackStruct l1CB)
{

	EA_LOG_VERBOSE("@@@ L1CallbackStruct @@@ \n");
	EA_LOG_VERBOSE("head_rect.x_start %f \n",	l1CB.head_rect.x_start);
	EA_LOG_VERBOSE("head_rect.y_start %f \n",	l1CB.head_rect.y_start);
	EA_LOG_VERBOSE("head_rect.x_end %f \n",		l1CB.head_rect.x_end);
	EA_LOG_VERBOSE("head_rect.y_end %f \n",		l1CB.head_rect.y_end);
	EA_LOG_VERBOSE("is_driver_valid %d \n",		l1CB.is_driver_valid);
	EA_LOG_VERBOSE("@@@ L1CallbackStruct @@@ \n");
	
	EA_LOG_VERBOSE("MaleFemal %d \n",		l1CB.ATTR_zip.MaleFemale);
	EA_LOG_VERBOSE("mas %d \n",				l1CB.ATTR_zip.mask);
	EA_LOG_VERBOSE("glasses %d \n",			l1CB.ATTR_zip.glasses);
	EA_LOG_VERBOSE("age_YMO %d \n",			l1CB.ATTR_zip.age_YMO);
	EA_LOG_VERBOSE("action_SP %d \n",		l1CB.ATTR_zip.action_SP);


	if (EA_LOG_GET_LOCAL() > 2)
	{
		//face
		ea_display_obj_params(display)->obj_win_w = 1.0;
		ea_display_obj_params(display)->obj_win_h = 1.0;

		ea_display_obj_params(display)->border_thickness = 5;
		ea_display_obj_params(display)->font_size = 32;
		ea_display_obj_params(display)->text_color = EA_16_COLORS_RED;
		ea_display_obj_params(display)->box_color = EA_16_COLORS_RED;

		if (l1CB.is_driver_valid == 1)
			ea_display_set_bbox(display, "TARGET", // 0~1 no pixel
								l1CB.head_rect.x_start,
								l1CB.head_rect.y_start,
								l1CB.head_rect.x_end - l1CB.head_rect.x_start,
								l1CB.head_rect.y_end - l1CB.head_rect.y_start);

		//landmark
		ea_display_obj_params(display)->font_size = 24;
		ea_display_obj_params(display)->border_thickness = 0;

		float x = 0.01;
		float y = 0;
		float w = 0.001;
		float h = 0.001;

		displayHandler(l1CB.ATTR_zip.MaleFemale, "Gender", gender_text_map, y, x, w, h);
		displayHandler(l1CB.ATTR_zip.mask, "Mask", mask_text_map, y, x, w, h);
		displayHandler(l1CB.ATTR_zip.glasses, "Glasses", glasses_text_map, y, x, w, h);
		displayHandler(l1CB.ATTR_zip.age_YMO, "Age", age_YMO_text_map, y, x, w, h);
		displayHandler(l1CB.ATTR_zip.action_SP, "Action", action_SP_text_map, y, x, w, h);


		ea_display_obj_params(display)->border_thickness = 10;
		ea_display_obj_params(display)->box_color = EA_16_COLORS_WHITE;
		if (l1CB.is_driver_valid == 1)
		for (int i = 0; i < 68; i++)
			ea_display_set_bbox(display, NULL, l1CB.landmark_img[i*2], l1CB.landmark_img[i*2+1] , 0.001, 0.001);


	}

	
}
