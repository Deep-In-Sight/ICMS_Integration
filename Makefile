Current_Folder :=$(CURDIR)
Current_src_Folder =$(CURDIR)/src
FIX_SO_DIR=$(CURDIR)/so/ori_so


$(shell rm -r $(Current_src_Folder)/result_O)
$(shell rm  $(Current_Folder)/test_eazyai)



AMB_TOPDIR=/home/umu/Desktop/0_minu/0_Amba/file/cv22iccr/ambarella
PREBUILD_OSS_DIR=$(AMB_TOPDIR)/prebuild/oss/armv8-a


LOCAL_TARGET	:= test_eazyai
LOCAL_SRCS	:= $(Current_src_Folder)/test_eazyai.cpp \
$(Current_src_Folder)/ICMSparser.c\
$(Current_src_Folder)/DinsightICC.cpp\
$(Current_src_Folder)/landmark.cpp\
$(Current_src_Folder)/face.cpp\
$(Current_src_Folder)/gaze.cpp\


LOCAL_CFLAGS  = -Wall -O3 -fopenmp \
		-I$(AMB_TOPDIR)/packages/eazyai/inc \
		-I$(PREBUILD_OSS_DIR)/lua/include \
		-I$(PREBUILD_OSS_DIR)/opencv/include/opencv4 \
	 	-I$(Current_src_Folder)

LOCAL_CFLAGS    += -DEIGEN_MPL2_ONLY  # For Eigen library to use MPL2 license related part only


LOCAL_LDFLAGS   := -L$(PREBUILD_OSS_DIR)/lua/usr/lib -llua

LOCAL_LIBS		:= libeazyai.so libeazyai_utils.so libeazyai_comm.so 


OPENCV_PATH		:= $(PREBUILD_OSS_DIR)/opencv


LOCAL_CFLAGS	+= -I$(OPENCV_PATH)/include/opencv4

LOCAL_LDFLAGS	+= -lpthread -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -lopencv_videoio  \
	-L$(OPENCV_PATH)/usr/lib \
	-Wl,-rpath-link=$(FIX_SO_DIR)/usr/lib \
	-Wl,-rpath-link=$(OPENCV_PATH)/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/libjpeg-turbo/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/tbb/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/freetype/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/libpng/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/zlib/usr/lib \
	-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/bzip2/usr/lib \
	-lstdc++

AMBARELLA_APP_CFLAGS = -I$(AMB_TOPDIR)/boards/cv22_walnut -I$(AMB_TOPDIR)/include -Wformat -Werror=format-security -I$(AMB_TOPDIR)/include/arch_v5 -Wall -D_REENTRENT -D_GNU_SOURCE -fasynchronous-unwind-tables -O3  -fdiagnostics-color=auto -march=armv8-a+crypto -mlittle-endian -mcpu=cortex-a53+crypto --param l1-cache-line-size=64 --param l1-cache-size=32 -Wp,-D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection
AMBARELLA_APP_LDFLAGS = -Wl,-as-needed -march=armv8-a+crypto -mlittle-endian -mcpu=cortex-a53+crypto --param l1-cache-line-size=64 --param l1-cache-size=32 -Wl,-z,relro,-z,now
AMBA_MAKEFILE_V		:= @

LOCAL_OBJS = $(patsubst $(Current_src_Folder)/%.c,$(Current_src_Folder)/result_O/%.o,$(filter %.c,$(LOCAL_SRCS))) $(patsubst $(Current_src_Folder)/%.cpp,$(Current_src_Folder)/result_O/%.o,$(filter %.cpp,$(LOCAL_SRCS)))

.PHONY: $(LOCAL_TARGET)

LOCAL_MODULE	:= $(LOCAL_TARGET)


__AMB_LIBS__	:= $(patsubst lib%.so, -l%, $(filter %.so,$(LOCAL_LIBS)))

$(LOCAL_MODULE): PRIVATE_AMB_LIBS := $(LOCAL_LIBS) #so file
$(LOCAL_MODULE): PRIVATE_CFLAGS := $(LOCAL_CFLAGS) #header file
$(LOCAL_MODULE): PRIVATE_LDFLAGS := $(__AMB_LIBS__) $(LOCAL_LDFLAGS)




$(Current_src_Folder)/result_O/%.o: $(Current_src_Folder)/%.c
	@echo "echo : compile_DOLLAR_AT         $@"
	@mkdir -p $(dir $@)
	$(AMBA_MAKEFILE_V) /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4/bin/aarch64-linux-gnu-g++ $(AMBARELLA_APP_CFLAGS) $(PRIVATE_CFLAGS) -MMD -c $< -o $@
$(info here?")
$(Current_src_Folder)/result_O/%.o: $(Current_src_Folder)/%.cpp 
	@mkdir -p $(dir $@)
	$(AMBA_MAKEFILE_V) /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4/bin/aarch64-linux-gnu-g++ $(AMBARELLA_APP_CFLAGS) $(PRIVATE_CFLAGS) -MMD -c $< -o $@




$(LOCAL_MODULE): $(LOCAL_OBJS) $(filter %.a, $(LOCAL_SRCS))  $(SO_FILES) 
	@echo "  "
	@echo "PRIVATE_LDFLAGS   $(PRIVATE_LDFLAGS)"
	@echo "  "
	@echo "  "
	@echo "LOCAL_LDFLAGS $(LOCAL_LDFLAGS)"
	$(AMBA_MAKEFILE_V) /usr/local/linaro-aarch64-2020.09-gcc10.2-linux5.4/bin/aarch64-linux-gnu-g++ $(AMBARELLA_APP_LDFLAGS)  \
		-L$(Current_Folder)/so\
		-o $@ $(filter-out %.a %.so %.lds %make.inc, $^)  $(PRIVATE_LDFLAGS) -lopencv_core -lopencv_imgproc -llua
	@echo "minu1minu Build $@ Done."
	@mv -f $(Current_Folder)/test_eazyai /home/linux_share/test_eazyai

