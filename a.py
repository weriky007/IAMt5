import MetaTrader5 as mt5
import pandas as pd

symbol = 'AUDCAD'


def fechar_ordem(ticket, symbol, quantidade, tipo_ordem):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Não foi possível obter cotações para {symbol}")
            return False

        # Prepara a requisição de fechamento
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": quantidade,
            "deviation": 5,
            "magic": 0,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "price": tick.bid if tipo_ordem == mt5.ORDER_TYPE_BUY else tick.ask,
            "type": mt5.ORDER_TYPE_SELL if tipo_ordem == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        }

        resultado = mt5.order_send(request)

        if resultado.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erro ao fechar ordem {ticket}: {resultado.comment}")
            return False

        print(f"✅ Ordem {ticket} fechada com sucesso!")
        return True

    except Exception as e:
        print(f"Erro crítico: {str(e)}")
        return False


def fechar_todas_ordens():
    try:
        posicoes = mt5.positions_get(symbol=symbol)

        if posicoes is None or len(posicoes) == 0:
            print(f"Nenhuma posição aberta encontrada para {symbol}")
            return

        print(f"\n🔍 Encontradas {len(posicoes)} posições abertas para {symbol}")

        for posicao in posicoes:
            print("\n" + "-" * 50)
            print(f"Ticket: {posicao.ticket}")
            print(f"Tipo: {'COMPRA' if posicao.type == 0 else 'VENDA'}")
            print(f"Volume: {posicao.volume}")
            print(f"Aberto em: {pd.to_datetime(posicao.time, unit='s')}")  # Corrigido com pandas
            print(f"Preço de abertura: {posicao.price_open:.5f}")
            print(f"Lucro atual: {posicao.profit:.2f} USD")

            if not fechar_ordem(
                    ticket=posicao.ticket,
                    symbol=symbol,
                    quantidade=posicao.volume,
                    tipo_ordem=posicao.type
            ):
                continue

            for _ in range(3):
                if not mt5.positions_get(ticket=posicao.ticket):
                    break
                mt5.sleep(300)
            else:
                print(f"⚠️ Aviso: Posição {posicao.ticket} ainda aparece como aberta")

    except Exception as e:
        print(f"Erro ao processar posições: {str(e)}")

if __name__ == "__main__":
    # Configuração inicial
    if not mt5.initialize():
        print("❌ Falha na conexão com o MetaTrader 5")
        exit(1)

    print(f"\n🚀 Iniciando fechamento de ordens para {symbol}...")
    fechar_todas_ordens()

    mt5.shutdown()
    print("\n✅ Operação concluída com sucesso!")