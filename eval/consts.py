CONTEXTS = 'contexts'
RESPONSES = 'responses'
REFERENCES = 'references'

# Bleu
LIST_OF_REFERENCES = 'list_of_references'

# EmbeddingBaseScore
EMBEDDINGS = 'embeddings'

# LSDSCC
HYPOTHESIS_SETS = 'hypothesis_sets'
REFERENCE_SETS = 'reference_sets'

# Adem
ADEM_MODEL = 'adem_model'
RAW_CONTEXTS = 'raw_contexts'
RAW_RESPONSES = 'raw_responses'
RAW_REFERENCES = 'raw_references'

# default embeddings for EB.
GOOGLE_NEWS_300_BIN = '/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin'

# The name of the dump of config.
CONFIG_JSON = 'config.json'

# The char that separates different params: model, dataset and metric.
SEPARATOR = '-'


# Correlation methods in pandas names.
PEARSON = 'pearson'
KENDALL = 'kendall'
SPEARMAN = 'spearman'

ALL_METHODS = (PEARSON, KENDALL, SPEARMAN)

# transparency of scatter plot.
SCATTER_ALPHA = 0.5

SERBAN_UBUNTU_MODEL_DIR = '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/ModelPredictions'

SERBAN_TWITTER_MODEL_DIR = '/home/cgsdfc/TwitterDialogueCorpus/ModelResponses'

# for ADEM
ADEM_ROOT = '/home/cgsdfc/deployment/Metrics/AutoTuring/ADEM-1-master'
ADEM_OUTPUT_FILE = 'adem_output.txt'
ADEM_IMAGE = 'cgsdfc/adem-1-master:latest'

# for meteor
METEOR_JAR_FILE = '/home/cgsdfc/deployment/Metrics/METEOR/meteor-1.5/meteor-1.5.jar'

# lsdscc
MULTI_RESPONSES = 'multi_responses'

# serban ppl
SERBAN_MODEL_ROOT = '/home/cgsdfc/SavedModels/HRED-VHRED'
SERBAN_RESULT_ROOT = '/home/cgsdfc/Result/HRED-VHRED'

TEST_DIALOGUES = 'test_dialogues'

MODEL_WEIGHTS = 'model_weights'
