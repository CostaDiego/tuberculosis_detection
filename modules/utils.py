def get_2heads_criterion(criterion):
    def _criterion(y_pred, y_true):
        abnormal_loss = criterion(y_pred[0], y_true[0])
        tuberculosis_loss = criterion(y_pred[1], y_true[1])
        return abnormal_loss + tuberculosis_loss

    return _criterion
