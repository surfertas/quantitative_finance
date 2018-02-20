#!/usr/bin/env python
# @author Tasuku Miura

import pandas
import talib as ta


class TAFeatures(object):
    """
    Technical indicators using talib.
    http://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
    Note: 
    The class is not flexible, ie, no API to select subset of desired
    features. Most indicators were added as is, as objective was to work
    on as many price indicators as possible for experimination.
    """
    def __init__(self, df):
        """
        Args:
            df - a pandas data frame containing data from Quandl.
        """
        self._df = df
        self._df_shifted = df.shift(1)
        self._df_index = self._df_shifted.index
        self._set_ohlc(self._df_shifted)
    
    def _set_ohlc(self, df):
        """ Sets the OHLC to be used for computing indicators.
        Args:
            df - a pandas df. Should be lagged to avoid look ahead.
        """
        self._o = np.array(df['Adj. Open'], dtype='f8') 
        self._h = np.array(df['Adj. High'], dtype='f8') 
        self._l = np.array(df['Adj. Low'], dtype='f8')
        self._c = np.array(df['Adj. Close'], dtype='f8')

    def get_features(self):
        """ Gets TA related to momenutm, vol, candles, cycles, ohlc.
        Returns:
            df_final - pandas df with all TA data stored in columns.
        """
        # Get log returns from open to close on same day.
        log_ret_df = self.get_open_to_close_log_returns(df[['Adj. Close','Adj. Open']])

        # Get features
        momentum_df = get_momentum_indicators()
        hist_vol_df = get_hist_vol_indicators()
        pattern_df = get_pattern_recognition_indicators()
        cycle_df = get_cycle_indicators()
        overlap_df = get_overlap_indicators()
        ohlc_df = get_ohlc_features()
        
        dfs = [
            self._df_shifted,
            log_ret_df,
            momentum_df,
            hist_vol_df,
            pattern_df,
            cycle_df,
            overlap_df,
            ohlc_df
        ]
        df_final = reduce(
            lambda l, r: pd.merge(l, r, left_index=True, right_index=True),
            dfs
        ).dropna()
        return df_final


    def get_close_to_close_log_returns(self, df):
        """ Computes close to close returns.
        Args:
            df - pandas df.
        Returns:
            df_log - dataframe with log returns.
        Note: Not used in this excercise.
        """
        df_log = pd.DataFrame(np.log(df/df.shift(1))).dropna()
        df_log = df_log.rename(index=str, columns={"Adj. Close": "log_ret"})
        return df_log

    def get_open_to_close_log_returns(self, df):
         """ Computes open to close returns.
        Args:
            df - pandas df.
        Returns:
            df - dataframe with log returns.
        """
        df['log_ret'] = np.log(df['Adj. Close']/df['Adj. Open'])
        del df['Adj. Close']
        del df['Adj. Open']
        return df

    def get_momentum_indicators(self, days=[7,14,28, 56]):
        """ Calculates momentum based indicators.
        Args:
            days - list of time intervals.
        Returns:
            df - pandas dataframe.
        """
        momentum = dict()
        for t in days:
            momentum['adx_{}'.format(t)] = ta.ADX(self._h, self._l, self._c, timeperiod=t)
            momentum['adxr_{}'.format(t)] = ta.ADXR(self._h, self._l, self._c, timeperiod=t)
            momentum['aroondown_{}'.format(t)], momentum['aroonup_{}'.format(t)] = ta.AROON(self._h, self._l, timeperiod=t)
            momentum['aroon_{}'.format(t)] = ta.AROONOSC(self._h, self._l, timeperiod=t)
            momentum['rsi_{}'.format(t)] = ta.RSI(self._c, timeperiod=t)
            momentum['mom_{}'.format(t)] = ta.MOM(self._c, timeperiod=t)
            momentum['roc_{}'.format(t)] = ta.ROC(self._c, timeperiod=t)
            momentum['willr_{}'.format(t)] = ta.WILLR(self._h, self._l, self._c, timeperiod=t)
            momentum['trix_{}'.format(t)] = ta.TRIX(self._c, timeperiod=t)
            
        momentum['apo'] = ta.APO(self._c, fastperiod=12, slowperiod=26, matype=0)
        momentum['macd'], momentum['macdsignal'], momentum['macdhist'] = ta.MACD(
            self._c, fastperiod=12, slowperiod=26, signalperiod=9)
        momentum['ppo'] = ta.PPO(self._c, fastperiod=12, slowperiod=26, matype=0)
        momentum['sself._lk'], momentum['sself._ld'] = ta.STOCH(
            self._h, self._l, self._c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        momentum['fastk'], momentum['fastd'] = ta.STOCHF(
            self._h, self._l, self._c, fastk_period=5, fastd_period=3, fastd_matype=0)
        momentum['fastkrsi'], momentum['fastdrsi'] = ta.STOCHRSI(
            self._c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        
        df = pd.DataFrame.from_dict(momentum)
        df = df.set_index(self._df_index)
        return df

    def get_hist_vol_indicators(self, days=[7,14,28, 56]):
        """ Calculates hist vol indicators.
        Args:
            days - list of time intervals.
        Returns:
            df - pandas dataframe.
        """
        hist_vol = dict()
        for t in days:
            hist_vol['atr_{}'.format(t)] = ta.ATR(self._h, self._l, self._c, timeperiod=t)
            hist_vol['natr_{}'.format(t)] = ta.NATR(self._h, self._l, self._c, timeperiod=t)
            hist_vol['std_{}'.format(t)] = ta.STDDEV(self._c, timeperiod=t, nbdev=1)
            hist_vol['var_{}'.format(t)] = ta.VAR(self._c, timeperiod=t, nbdev=1)
            hist_vol['linreg_{}'.format(t)] = ta.LINEARREG(self._c, timeperiod=t)

        hist_vol['trange'] = ta.TRANGE(self._h, self._l, self._c)
        df = pd.DataFrame.from_dict(hist_vol)
        df = df.set_index(self._df_index)
        return df
    
    def get_pattern_recognition_indicators(self):
        """ Calculates candle indicators.
        Returns:
            df - pandas dataframe.
        """
        patterns = dict()
        patterns['CDL2CROWS'] = ta.CDL2CROWS(self._o, self._h, self._l, self._c)
        patterns['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(self._o, self._h, self._l, self._c)
        patterns['CDL3INSIDE'] = ta.CDL3INSIDE(self._o, self._h, self._l, self._c)
        patterns['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(self._o, self._h, self._l, self._c)
        patterns['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(self._o, self._h, self._l, self._c)
        patterns['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(self._o, self._h, self._l, self._c)
        patterns['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(self._o, self._h, self._l, self._c)
        patterns['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(self._o, self._h, self._l, self._c)
        patterns['CDLBELTHOLD'] = ta.CDLBELTHOLD(self._o, self._h, self._l, self._c)
        patterns['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(self._o, self._h, self._l, self._c)
        patterns['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(self._o, self._h, self._l, self._c)
        patterns['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(self._o, self._h, self._l, self._c)
        patterns['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(self._o, self._h, self._l, self._c)
        patterns['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLDOJI'] = ta.CDLDOJI(self._o, self._h, self._l, self._c)
        patterns['CDLDOJISTAR'] = ta.CDLDOJISTAR(self._o, self._h, self._l, self._c)
        patterns['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(self._o, self._h, self._l, self._c)
        patterns['CDLENGULFING'] = ta.CDLENGULFING(self._o, self._h, self._l, self._c)
        patterns['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(self._o, self._h, self._l, self._c)
        patterns['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(self._o, self._h, self._l, self._c)
        patterns['CDLHAMMER'] = ta.CDLHAMMER(self._o, self._h, self._l, self._c)
        patterns['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(self._o, self._h, self._l, self._c)
        patterns['CDLHARAMI'] = ta.CDLHARAMI(self._o, self._h, self._l, self._c)
        patterns['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(self._o, self._h, self._l, self._c)
        patterns['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(self._o, self._h, self._l, self._c)
        patterns['CDLHIKKAKE'] = ta.CDLHIKKAKE(self._o, self._h, self._l, self._c)
        patterns['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(self._o, self._h, self._l, self._c)
        patterns['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(self._o, self._h, self._l, self._c)
        patterns['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(self._o, self._h, self._l, self._c)
        patterns['CDLINNECK'] = ta.CDLINNECK(self._o, self._h, self._l, self._c)
        patterns['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(self._o, self._h, self._l, self._c)
        patterns['CDLKICKING'] = ta.CDLKICKING(self._o, self._h, self._l, self._c)
        patterns['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(self._o, self._h, self._l, self._c)
        patterns['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(self._o, self._h, self._l, self._c)
        patterns['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(self._o, self._h, self._l, self._c)
        patterns['CDLLONGLINE'] = ta.CDLLONGLINE(self._o, self._h, self._l, self._c)
        patterns['CDLMARUBOZU'] = ta.CDLMARUBOZU(self._o, self._h, self._l, self._c)
        patterns['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(self._o, self._h, self._l, self._c)
        patterns['CDLMATHOLD'] = ta.CDLMATHOLD(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(self._o, self._h, self._l, self._c, penetration=0)
        patterns['CDLONNECK'] = ta.CDLONNECK(self._o, self._h, self._l, self._c)
        patterns['CDLPIERCING'] = ta.CDLPIERCING(self._o, self._h, self._l, self._c)
        patterns['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(self._o, self._h, self._l, self._c)
        patterns['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(self._o, self._h, self._l, self._c)
        patterns['CDLSEPARATINGLINE'] = ta.CDLSEPARATINGLINES(self._o, self._h, self._l, self._c)
        patterns['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(self._o, self._h, self._l, self._c)
        patterns['CDLSHORTLINE'] = ta.CDLSHORTLINE(self._o, self._h, self._l, self._c)
        patterns['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(self._o, self._h, self._l, self._c)
        patterns['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(self._o, self._h, self._l, self._c)
        patterns['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(self._o, self._h, self._l, self._c)
        patterns['CDLTAKURI'] = ta.CDLTAKURI(self._o, self._h, self._l, self._c)
        patterns['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(self._o, self._h, self._l, self._c)
        patterns['CDLTHRUSTING'] = ta.CDLTHRUSTING(self._o, self._h, self._l, self._c)
        patterns['CDLTRISTAR'] = ta.CDLTRISTAR(self._o, self._h, self._l, self._c)
        patterns['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(self._o, self._h, self._l, self._c)
        patterns['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(self._o, self._h, self._l, self._c)
        patterns['DLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(self._o, self._h, self._l, self._c)
        df = pd.DataFrame.from_dict(patterns)
        df = df.set_index(self._df_index)
        return df

    def get_cycle_indicators(self):
        """ Calculates cycle indicators.
        Returns:
            df - pandas dataframe.
        """
        cycle = dict()
        cycle['HT_DCPERIOD'] = ta.HT_DCPERIOD(self._c)
        cycle['HT_DCPHASE'] = ta.HT_DCPHASE(self._c)
        cycle['HT_DCPERIOD'] = ta.HT_DCPERIOD(self._c)
        cycle['inphase'], cycle['quadrature'] = ta.HT_PHASOR(self._c)
        cycle['sine'], cycle['leadsine'] = ta.HT_SINE(self._c)
        cycle['integer'] = ta.HT_TRENDMODE(self._c)
        df = pd.DataFrame.from_dict(cycle)
        df = df.set_index(self._df_index)
        return df

    def get_overlap_indicators(self, days=[7, 14, 28, 56]):
        """ Calculates overlap indicators.
        Args:
            days - list of time intervals.
        Returns:
            df - pandas dataframe.
        """
        overlap = dict()
        for t in days:
            bbands = ta.BBANDS(self._c, timeperiod=t, nbdevup=2, nbdevdn=2, matype=0)
            overlap['bb_upperband_{}'.format(t)], overlap['bb_middleband_{}'.format(t)], overlap['bb_lowerband_{}'.format(t)] = bbands
            overlap['DEMA_{}'.format(t)] = ta.DEMA(self._c, timeperiod=t)
            overlap['EMA_{}'.format(t)] = ta.EMA(self._c, timeperiod=t)
            overlap['KAMA_{}'.format(t)] = ta.KAMA(self._c, timeperiod=t)
            overlap['MA_{}'.format(t)] = ta.MA(self._c, timeperiod=t, matype=0)
            overlap['MIDPOINT_{}'.format(t)] = ta.MIDPOINT(self._c, timeperiod=t)
            overlap['MIDPRICE_{}'.format(t)] = ta.MIDPRICE(self._h, self._l, timeperiod=t)
            overlap['SMA_{}'.format(t)] = ta.SMA(self._c, timeperiod=t)
            overlap['T3_{}'.format(t)] = ta.T3(self._c, timeperiod=t, vfactor=0)
            overlap['TEMA(_{}'.format(t)] = ta.TEMA(self._c, timeperiod=t)
            overlap['TRIMA_{}'.format(t)] = ta.TRIMA(self._c, timeperiod=t)
            overlap['WMA_{}'.format(t)] = ta.WMA(self._c, timeperiod=t)
            overlap['HT_TRENDLINE'] = ta.HT_TRENDLINE(self._c)
            overlap['mama'], overlap['fama'] = ta.MAMA(self._c, fastlimit=0.9, slowlimit=0.1)
            overlap['SAR'] = ta.SAR(self._h, self._l, acceleration=0, maximum=0)
            overlap['SAREXT'] = ta.SAREXT(
                self._h, self._l, startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0,
                accelerationshort=0, accelerationmaxshort=0
            )
        df = pd.DataFrame.from_dict(overlap)
        df = df.set_index(self._df_index)
        return df

    def get_ohlc_features(self):
       """ Calculates olhc related indicators.
        Returns:
            df - pandas dataframe.
        """
 
        olhc = dict()
        olhc['hilo_diff'] = self._h - self._l
        olhc['opcl_diff'] = self._c - self._o
        df = pd.DataFrame.from_dict(olhc)
        df = df.set_index(self._df_index)
        return df
