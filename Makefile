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
.PHONY: help build clean convert train

help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Targets:'
	@echo '  build    	build docker $(IMG) for current user: $(HOST_USER)(uid=$(HOST_UID))'
	@echo '  clean    	remove docker image $(IMG) for current user: $(HOST_USER)(uid=$(HOST_UID))'
	@echo '  convert_16	convert frozen tensorflow model to openvino FP16 format for Neural Compute Stick'
	@echo '  convert_32 convert frozen tensorflow model to openvino FP32 format for desktop'
	@echo '  prune    	shortcut for docker system prune -af. Cleanup inactive containers and cache.'
	@echo '  rebuild  	rebuild docker $(IMG) for current user: $(HOST_USER)(uid=$(HOST_UID))'
	@echo '  shell		run docker --container-- for current user: $(HOST_USER)(uid=$(HOST_UID))'
	@echo '  train		train the model on data in the data/train directory'
	@echo ''

build:
	docker build -t $(IMG) .

clean:
	docker stop $(APP); docker rm $(APP)

convert_16:
	docker run -u $(HOST_UID):$(HOST_GID) -it -d --mount type=bind,source=${CURDIR}/models,destination=/app/models --name $(APP) $(IMG);
	docker exec -w /app/models/openvino $(APP) python $(MO_TF) --input_model /app/models/$(MOD).pb -b $(BATCH_SIZE) --data_type FP16 --scale 255 --reverse_input_channels;
	docker stop $(APP)
	docker rm $(APP)


convert_32:
	docker run -u $(HOST_UID):$(HOST_GID) -it -d --mount type=bind,source=${CURDIR}/models,destination=/app/models --name $(APP) $(IMG);
	docker exec -w /app/models/openvino $(APP) python $(MO_TF) --input_model /app/models/$(MOD).pb -b $(BATCH_SIZE) --data_type FP32 --scale 255 --reverse_input_channels;
	docker stop $(APP)
	docker rm $(APP)

shell:
	docker run -it --mount type=bind,source=${CURDIR}/data,destination=/app/data,readonly --name $(APP) $(IMG);

train:
	docker run -u $(HOST_UID):$(HOST_GID) -it -d --mount type=bind,source=${CURDIR}/data,destination=/app/data,readonly,consistency=cached --name $(APP) $(IMG);
	docker exec $(APP) python src/tr_image.py data/train;
	docker cp $(APP):/app/models/. ${CURDIR}/models/;
	docker stop $(APP)
	docker rm $(APP)