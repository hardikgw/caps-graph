import os
import Cayley
from google.cloud import language_v1
from google.cloud.language_v1 import enums
import re


def get_entities(text_content, filename):
    """
    Analyzing Entities in a String

    Args:
      text_content The text content to analyze
    """
    # Open Text file for output
    f = open("data/entities.nq", "a+")

    client = language_v1.LanguageServiceClient()

    # text_content = 'California is a state.'

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_entities(document, encoding_type=encoding_type)
    # Loop through entitites returned from the API
    for entity in response.entities:

        # Write to text file
        entityType = enums.Entity.Type(entity.type).name
        if not entityType == "NUMBER":
            f.write(
                u"<{}> <{}> <{}> <{}> .\n".format(re.sub(' +', '_', entity.name), "mentioned_in", filename, entityType))
        print("---------------------->>>>>>>>>><<<<<<<<<<-------------------------")
        print(u"Representative name for the entity: {}".format(entity.name))
        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        print(u"Entity type: {}".format(enums.Entity.Type(entity.type).name))
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(entity.salience))
        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        for metadata_name, metadata_value in entity.metadata.items():
            print(u"{}: {}".format(metadata_name, metadata_value))

        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        for mention in entity.mentions:
            print(u"Mention text: {}".format(mention.text.content))
            # Get the mention type, e.g. PROPER for proper noun
            print(
                u"Mention type: {}".format(enums.EntityMention.Type(mention.type).name)
            )

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(u"Language of the text: {}".format(response.language))


def cayley_import(_subject, _predicate, _object, _label):
    import requests
    import json

    url = "http://localhost:64210/api/v1/write"
    quad = [{
        "subject": _subject,
        "predicate": _predicate,
        "object": _object,
        "label": _label
    }]
    x = requests.post(url, data=json.dumps(quad))
    print(x.text)


os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/hardikpatel/workbench/projects/cit/caps-graph/data/keys/NER-Ocean-ce681797e70f.json"

# cayley_import("jim", "mentioned_in", "super magazine", "book")

data_folder = '/Users/hardikpatel/workbench/projects/cit/caps-graph/data/ocean/abstract/'
for i in os.listdir(data_folder):
    if i.endswith('.txt'):
        print("Processing File -----------> " + i)
        fo = open(data_folder + i, "rb")
        full_str = fo.read().decode(errors='replace')
        full_str_clean = full_str.encode('ascii', errors='ignore')
        get_entities(full_str_clean, i)
