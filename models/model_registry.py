model_hub = {}
aux_models = []
detail_head_models = []


def register_model(*other_registries):
    def decorator(model_class):
        model_hub[model_class.__name__.lower()] = model_class

        for registry in other_registries:
            if isinstance(registry, list):
                registry.append(model_class.__name__.lower())
            else:
                print(f"Model registry is not a list. Skipping registry: {registry}")

        return model_class
    return decorator