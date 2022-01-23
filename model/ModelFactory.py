import importlib


class ModelFactory(object):
    def __init__(self):
        pass
        self.model_path = ["model.item_ranking",
                           "model.seq_ranking"]

    def get_model(self, config):
        recommender = config["model"]
        for model_path in self.model_path:
            spec_path = "{}.{}".format(model_path, recommender)
            if importlib.util.find_spec(spec_path):
                module = importlib.import_module(spec_path)
                break
        else:
            raise ImportError("Recommender: {} not found".format(recommender))
        if hasattr(module, recommender):
            Recommender = getattr(module, recommender)
        else:
            raise ImportError("Import '%s' failed from '%s'!" % (recommender, module.__file__))
        model = Recommender(config)
        return model
