.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROJECT_NAME = Synopsys Project 2016
PYTHON_INTERPRETER = python
TENSORBOARD = tensorboard
IS_ANACONDA=$(shell python -c "import sys;t=str('anaconda' in sys.version.lower() or 'continuum' in sys.version.lower());sys.stdout.write(t)")
MODELS = src/models
VISUALIZATION = src/visualization
LOGS_DIR = models/run_logs
MODEL_NAME = ${MODEL}
# MODEL_FUNC = ${MODEL_FUNC}
MODEL_ARGS_STR = ${MODEL_ARGS}
ADDITIONAL_ARGS = ${ARGS}
WEIGHTS_DIR = models/weights
JSON_DIR = models/json
YAML_DIR = models/yaml
CSV_DIR = models/csv
IMAGES_DIR = reports/figures
TENSORBOARD_DIR = models/run_logs/tensorboard
UNIT_TESTS = src/unit_tests/*.py

# Only writes to txt file if the argument "--test-only" is not set
PRINT_TO_TXT= if !(echo $(ADDITIONAL_ARGS) | grep -q "test-only"); then \
                  cat .tmp | col -b >> "$(LOGS_DIR)/$(MODEL_NAME).txt"; \
              fi

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Upload Data to S3
sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

## Download Data from S3
sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

## Set up python interpreter environment
create_environment:
ifeq (True,$(IS_ANACONDA))
		@echo ">>> Detected Anaconda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.5
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# USER DEFINED ENVIRONMENTS                                                     #
#################################################################################

PYTHON2.7_ENV:
	source activate python2.7

PYTHON3.5_ENV:
	workon python3.5;

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Interpret special characters like backspace (^H) with `col -b`
# and write (unbuffered) run output to a txt file with the specified model name
train: 
	@$(PYTHON_INTERPRETER) -u $(MODELS)/train_model.py \
		$(MODEL_NAME) \
		$(MODEL_FUNC) \
		$(WEIGHTS_DIR)/$(MODEL_NAME).hdf5 \
		$(JSON_DIR)/$(MODEL_NAME).json \
		$(YAML_DIR)/$(MODEL_NAME).yaml \
		$(CSV_DIR)/$(MODEL_NAME).csv \
		$(TENSORBOARD_DIR) \
		$(MODEL_ARGS_STR) \
		$(ADDITIONAL_ARGS) | tee .tmp; \
		$(PRINT_TO_TXT)

# TODO: Make testing and training be from the same file
test:
	@$(PYTHON_INTERPRETER) -u $(MODELS)/test_model.py \
		$(MODEL_NAME) \
		${MODEL_FUNC} \
		$(WEIGHTS_DIR)/$(MODEL_NAME).hdf5 \
		$(JSON_DIR)/$(MODEL_NAME).json \
		$(YAML_DIR)/$(MODEL_NAME).yaml \
		$(CSV_DIR)/$(MODEL_NAME).csv \
		$(TENSORBOARD_DIR) \
		$(MODEL_ARGS_STR) \
		$(ADDITIONAL_ARGS) | tee .tmp;

# train_conv_net: 
# 	MODEL_FUNC="conv_net" \
# 	$(MAKE) train \
# 	$(MAKE) plot_train_valid
#
# train_danq:
# 	MODEL_FUNC="DanQ" \
# 	$(MAKE) train \
# 	$(MAKE) plot_train_valid

# Execute all the python unit tests
unit_tests:
	@for f in $(UNIT_TESTS); do \
		python $$f; \
	done

# Reformat any badly formatted (but valid) JSON or YAML representations
# of a Keras model
resave_json_yaml:
	@$(PYTHON_INTERPRETER) $(MODELS)/resave_json_yaml.py \
		"$(JSON_DIR)/$(MODEL_NAME).json" \
		"$(YAML_DIR)/$(MODEL_NAME).yaml"

# TODO: set up tox

# Plot the training and validation losses and accuracies
plot_train_valid:
	@$(PYTHON_INTERPRETER) $(VISUALIZATION)/plot_train_valid.py \
		"$(CSV_DIR)/$(MODEL_NAME).csv" \
		"$(IMAGES_DIR)/losses/$(MODEL_NAME)_loss.png" \
		"$(IMAGES_DIR)/accuracies/$(MODEL_NAME)_acc.png"

tensorboard:
	$(TENSORBOARD) --logdir=$(LOGS_DIR)/tensorboard


#################################################################################
# Self Documenting Commands                                                                #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
