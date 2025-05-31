def suggest_model_specific(trial, model):
    d_model_choices = [8, 16, 32, 64, 128]
    d_ff_choices = [8, 16, 32, 64, 128]
    n_heads_choices = [1, 2, 4, 8]
    e_layers_range = (1, 8)
    d_layers_range = (1, 8)
    factor_range = (1, 3)
    patch_len_choices = [4, 8, 16, 32, 64]
    top_k_choices = [1, 3, 5, 10, 20]



    def suggest_d_model_and_heads():
        while True:
            d_model = trial.suggest_categorical('d_model', d_model_choices)
            n_heads = trial.suggest_categorical('n_heads', n_heads_choices)
            if d_model % n_heads == 0:
                return d_model, n_heads

    if model == 'PatchTST':
        d_model, n_heads = suggest_d_model_and_heads()
        return {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
            'patch_len': trial.suggest_categorical('patch_len', patch_len_choices)
        }

    elif model in ['LSTM', 'GRU']:
        return {
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_model': trial.suggest_categorical('d_model', d_ff_choices),
        }

    elif model == 'Nonstationary_Transformer':
        d_model, n_heads = suggest_d_model_and_heads()
        p_hidden_layers = trial.suggest_categorical('p_hidden_layers', [2, 3, 4])
        hidden_unit = trial.suggest_categorical('hidden_unit', [64, 128, 256, 512])
        p_hidden_dims = [hidden_unit] * p_hidden_layers

        return {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
            'p_hidden_dims': p_hidden_dims,
            'p_hidden_layers': p_hidden_layers,
            'top_k': trial.suggest_categorical('top_k', top_k_choices),
        }


    elif model in ["DLinear", "FEDformer", "TimeNet", "iTransformer"]:
        d_model, n_heads = suggest_d_model_and_heads()
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': trial.suggest_int('e_layers', *e_layers_range),
            'd_layers': trial.suggest_int('d_layers', *d_layers_range),
            'd_ff': trial.suggest_categorical('d_ff', d_ff_choices),
            'factor': trial.suggest_int('factor', *factor_range),
        }
        if model == "FEDformer":
            params['top_k'] = trial.suggest_categorical('top_k', top_k_choices)
        return params

    else:
        return {}  # fallback if unknown model
