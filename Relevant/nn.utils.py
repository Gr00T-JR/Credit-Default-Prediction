def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.
    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)