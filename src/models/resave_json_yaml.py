import yaml
import json
import sys
import argparse
from keras.models import Sequential, model_from_yaml, model_from_json
from src.models.keras_model_utils import get_model, save_model

def pretty_resave(json_file=None, yaml_file=None):
    """ Resave the JSON and/or YAML file(s) and properly format them """
    if (json_file is not None):
        try:
            model = get_model(json_file=json_file)
            model_json = model.to_json()
            save_model(model, json_file=json_file)
        except:
            print (json_file + ' is not formatted in the format Keras uses')
            print (sys.exc_info()[0])

    if (yaml_file is not None):
        try:
            model = get_model(yaml_file=yaml_file)
            model_yaml = model.to_yaml()
            save_model(model, yaml_file=yaml_file)
        except:
            print (yaml_file + ' is not formatted in the format Keras uses')
            print (sys.exc_info()[0])

def main(argv):
    print ('')
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Pretty save JSON and YAML files')

    parser.add_argument('json_file', metavar='JSON_FILE', help='The file (.json) to store the model\'s architecture in JSON')
    parser.add_argument('yaml_file', metavar='YAML_FILE', help='The file (.yaml) to store the model\'s architecture in YAML')
    args = parser.parse_args()

    pretty_resave(json_file=args.json_file, yaml_file=args.yaml_file)

if __name__  == '__main__':
  main(sys.argv[1:])
