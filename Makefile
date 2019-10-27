# ----------------------
# Makefile for Movrasten
# ----------------------

# container names
IMG = movrasten_img
APP = movrasten_app
# args for model
MOD = model
MO_TF = /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py
BATCH_SIZE = 1

# if vars not set specifially: try default to environment, else fixed value.
# strip to ensure spaces are removed in future editorial mistakes.
# tested to work consistently on popular Linux flavors and Mac.
ifeq ($(user),)
# USER retrieved from env, UID from shell.
HOST_USER ?= $(strip $(if $(USER),$(USER),nodummy))
HOST_UID ?= $(strip $(if $(shell id -u),$(shell id -u),4000))
HOST_GID ?= $(strip $(if $(shell id -g),$(shell id -g),4000))
else
# allow override by adding user= and/ or uid=  (lowercase!).
# uid= defaults to 0 if user= set (i.e. root).
HOST_USER = $(user)
HOST_UID = $(strip $(if $(uid),$(uid),0))
endif

# commands not to be confused with files
.PHONY: all

help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Targets:'
	@echo '  all 		build, train, convert_32, stop'
	@echo '  build    	build docker $(IMG)'
	@echo '  clean    	remove docker image $(IMG)'
	@echo '  convert_16	convert frozen tensorflow model to openvino FP16 format for Neural Compute Stick'
	@echo '  convert_32 convert frozen tensorflow model to openvino FP32 format for desktop'
	@echo '  infer		condut OpenVINO inference on CPU'
	@echo '  infer_16	conduct OpenVINO inference on NCS'
	@echo '  prune    	shortcut for docker system prune -af. Cleanup inactive containers and cache.'
	@echo '  rebuild  	force rebuild docker $(IMG) for with --no-cache'
	@echo '  run 		run docker $(IMG) as $(APP) for current user: $(HOST_USER)(uid=$(HOST_UID))'
	@echo '  shell		open interactive shell to stopped container $(APP) for current user'
	@echo '  stop		stop $(APP)'
	@echo '  train		train the model on data in the data/train directory'
	@echo ''

all: | build train convert_32 stop

build:
	sudo docker build -t $(IMG) .

clean:
	sudo docker rm -f $(APP);

convert_16:
	sudo docker exec -w /app/models/openvino $(APP) python $(MO_TF) --input_model /app/models/$(MOD).pb -b $(BATCH_SIZE) --data_type FP16 --scale 255 --reverse_input_channels;

convert_32:
	sudo docker exec -w /app/models/openvino $(APP) python $(MO_TF) --input_model /app/models/$(MOD).pb -b $(BATCH_SIZE) --data_type FP32 --scale 255 --reverse_input_channels;

prune:
	sudo docker system prune -af

rebuild:
	sudo docker build --no-cache -t $(IMG) .

run:
	sudo docker run -u $(HOST_UID):$(HOST_GID) -it -d --mount type=bind,source=${CURDIR},destination=/app,consistency=cached --name $(APP) $(IMG);

shell:
	sudo docker start -i $(APP)

stop:
	sudo docker stop $(APP)

train:
	sudo docker exec $(APP) python /app/src/tr_image.py /app/data/train;