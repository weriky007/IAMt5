def calculate_position_size(balance, risk_percent, stop_loss_pips):
    risk_amount = balance * risk_percent / 100
    position_size = risk_amount / (stop_loss_pips * 0.0001)  # Para pares de 4 casas decimais
    return position_size