import MetaTrader5 as mt5

def connect_mt5():
    if not mt5.initialize():
        print("Falha ao inicializar o MT5")
        mt5.shutdown()
        return False
    print("Conectado ao MT5 com sucesso!")
    return True

def disconnect_mt5():
    mt5.shutdown()
    print("Desconectado do MT5")