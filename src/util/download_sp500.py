# On using adj.close: https://blog.quandl.com/guide-to-stock-price-calculation
# On downloading SP data: https://ntguardian.wordpress.com/2017/10/24/getting-sp-500-stock-data-from-quandlgoogle-with-python/
import pandas as pd
import quandl

from time import sleep

def download_sp500(start, end):
    symbols_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
    symbols = list(symbols_table.loc[:, "Ticker symbol"])
    n_symbols = len(symbols)
    sp500 = None
    for i,s in enumerate(symbols):
        print("{}/{}".format(i,n_symbols))
        sleep(2)
        print("Processing: " + s + "...", end)
        try: 
            s_data = quandl.get("WIKI/" + s, start_date=start, end_date=end)
            s_data = s_data.loc[:, "Adj. Close"]
            s_data.name = s
            if s_data.shape[0] > 1:
                if sp500 is None:
                    sp500 = pd.DataFrame(s_data)
                else:
                    sp500 = pd.concat([sp500,s_data], axis=1)
                    print(" Got it! From", s_data.index[0], "to", s_data.index[-1])
            else:
                print(" Sorry, but not this one!")
        except Exception as e:
            print("Failed: {}".format(e))

    badsymbols = list(set(s) - set(sp500.columns))
    if len(badsymbols) > 0:
        print("There were", len(badsymbols), "symbols for which data could not be obtained.")
        print("They are:", ", ".join(badsymbols))

    return sp500
