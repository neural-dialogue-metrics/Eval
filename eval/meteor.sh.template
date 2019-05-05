#!/usr/bin/env bash

# This tells Meteor to score the file "test" against "reference", where test and reference are UTF-8 encoded files that contain one sentence per line. The "-l en" option tells Meteor to use settings for English. The -norm flag tells Meteor to apply language-specific text normalization before scoring. These are the ideal settings for which language-specific parameters are tuned.

REFERENCE={references}
RESPONSE={responses}
JAR_FILE={jar_file}

java -Xmx2G -jar $JAR_FILE $RESPONSE $REFERENCE -l en -norm
