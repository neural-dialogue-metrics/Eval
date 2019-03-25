import collections


class EvalInfo(collections.namedtuple('EvalInfo', ['metric', 'model', 'run'])):
    """
    Information about an evaluation.

    Fields:
        metric: name of a metric, such as BLEU-4, ROUGE-L.
        model: name of a model, such as HRED_Baseline.
        run: name of the *inference* run, such as UbuntuDialogueCorpus.
    """

    def get_filename(self, ext=None):
        """
        Encode the info into a filename.
        :return:
        """
        if ext is None:
            ext = 'txt'
        return '%s-%s-%s.%s' % (self.metric, self.model, self.dataset, ext)
