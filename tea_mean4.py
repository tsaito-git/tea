#coding: utf8
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import math
import pathlib
import itertools

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
pd.plotting.register_matplotlib_converters()

from tob import TOBdf as TOB

def files_grouper(x):
    return x.name[:16]
def sum_slm(x):
    return x.sum()/600
def sum_slm_cor(x):
    return x.where(x < 1.0,1.0).sum()/600
def count_over(x):
    return (x >= 1.0).astype(int).sum()
def double_rotation(x):
    Ux,Ux_spike = spike_remove(x.Ux)
    Uy,Uy_spike = spike_remove(x.Uy)
    Uz,Uz_spike = spike_remove(x.Uz)
    ax,ay,az = Ux.mean(),Uy.mean(),Uz.mean()
    a1 = math.sqrt(ax*ax+ay*ay)
    a2 = math.sqrt(ax*ax+ay*ay+az*az)
    u = (Ux*ax + Uy*ay + Uz*az)/a2
    w = (Uz*a1-(Ux*ax+Uy*ay)*az/a1)/a2
    aw_abs = w.abs().mean()
    au,aw = u.mean(),w.mean()
    uw = -(((u-au)*(w-aw)).mean())
    return pd.Series({'au':au,'aw':aw,'aw_abs':aw_abs,'ustar2':uw,'Ux_spike':Ux_spike,'Uy_spike':Uy_spike,'Uz_spike':Uz_spike})
def spike_remove(x):
    limit = 3.5*x.std()
    xmean = x.mean()
    limit_in = x.between(xmean-limit,xmean+limit)
    return x.where(limit_in,float('nan')).fillna(method='pad'),(~limit_in).sum()#spikes(>3.5*std) padded,spike count
def cal_eddy(f):
    pkl_fn = pathlib.Path('eddy')/f[0].with_suffix('.parquet').name
    if pkl_fn.is_file():
        return pd.read_parquet(pkl_fn)
    dfs = []
    df = None
    for ff in f:
        tmp = TOB(ff,tz='Asia/Tokyo').df.copy()
        if len(tmp) > 0 and 'w' in tmp.columns:
            tmp = tmp.rename(columns={'flow_upstream':'flow_updraft','flow_downstream':'flow_downdraft'})
            dfs.append(tmp)
    if len(dfs) > 0:
        df = pd.concat(dfs)
    if df is not None and len(df) > 0 and 'w' in df.columns:
#        print(','.join(ff.name for ff in f))
        df = df.assign(w_abs=df.w.abs())
        gpr = pd.Grouper(freq='30min',closed='right',label='right')
        r = df.groupby(gpr).filter(lambda x: len(x) > 2).groupby(gpr)
        aggr = {
            'flow_updraft':('flow_updraft',sum_slm),
            'up_count_over':('flow_updraft',count_over),
            'flow_downdraft':('flow_downdraft',sum_slm),
            'down_count_over':('flow_downdraft',count_over),
            'u':('u','mean'),'w':('w','mean'),'w_abs':('w_abs','mean'),'Ts':('Ts','mean')
        }
        if 'setting_point_updraft' in df.columns:
            aggr.update({'setting_updraft':('setting_point_updraft',sum_slm)})
            aggr.update({'setting_updraft_cor':('setting_point_updraft',sum_slm_cor)})
        if 'setting_point_downdraft' in df.columns:
            aggr.update({'setting_downdraft':('setting_point_downdraft',sum_slm)})
            aggr.update({'setting_downdraft_cor':('setting_point_downdraft',sum_slm_cor)})
        dfavg = r.agg(**aggr)
        # dfedy = df[['u','w']].groupby(gpr).filter(lambda x: len(x) > 2).groupby(gpr).transform(lambda x: x-x.mean())
        # ustar2 = dfedy.groupby(gpr).apply(lambda r: -((r.w*r.u).mean()) )
        dfedy_dr = r.apply(double_rotation)
        dfavg = dfavg.join(dfedy_dr)
        dfavg.to_parquet(pkl_fn)
        return dfavg
def cal_conc(f,co2label='CO2digi2',h2olabel='H2Odigi2',start='2000-01-01',filetype='tob'):
    if filetype == 'tob':
        tob1 = pd.concat([TOB(ff,tz='Asia/Tokyo').df for ff in f])
        # print(','.join(ff.name for ff in f))
        df = tob1.sort_index().query('@start <= index')
        if len(df) == 0 or (co2label not in df.columns):
            return None
        df = df.assign(co2_dry=df[co2label]/(1-df[h2olabel]/1000.0))
        # baselag = pd.Timedelta(seconds=15)
        baselag = pd.Timedelta(seconds=50)#2022-07-15 for all
        limit = 4.0
    elif filetype == 'pic':
        df = pd.concat([pd.read_csv(ff,index_col=0,parse_dates=[0]) for ff in f])
        if len(df) == 0 or (co2label not in df.columns):
            return None
        df.index = pd.DatetimeIndex(df.index).tz_convert('Asia/Tokyo')
        df = df.sort_index().query('@start <= index')
        df = df.assign(co2_dry=df[co2label])
        baselag = pd.Timedelta(seconds=150)
        limit = 1.5 if co2label.startswith('X12') else 0.15

    ret = []
    jit = pd.Timedelta(seconds=1)
    for n,gg in df.groupby(pd.Grouper(freq='30Min',closed='right',label='left')):
        g = gg.copy()
        lag = baselag
        # if filetype == 'tob':
        #     if n >= pd.Timestamp('2021-04-28T11:30',tz="Asia/Tokyo"):
        #         lag = pd.Timedelta(seconds=17,milliseconds=500) if co2label == 'CO2digi2' else pd.Timedelta(seconds=18)
        #     if n > pd.Timestamp('2021-11-30T10:00',tz="Asia/Tokyo"):
        #         lag = baselag + pd.Timedelta(seconds=10) #change flowrate from 0.75l/min to 0.5l/min @2021-11-30T10:00
        if (pd.Timestamp('2021-12-02T10:30',tz="Asia/Tokyo") <= n <= pd.Timestamp('2021-12-02T15:30',tz="Asia/Tokyo")) or (pd.Timestamp('2022-06-01T10:30',tz="Asia/Tokyo") <= n <= pd.Timestamp('2021-06-01T15:30',tz="Asia/Tokyo")):
            duration = pd.Timedelta(minutes=2,seconds=0 if filetype == 'pic' else 20)
            up2dn = pd.Timedelta(minutes=3,seconds=5)
        else:
            duration = pd.Timedelta(minutes=8 if filetype == 'pic' else 9,seconds=0)
            up2dn = pd.Timedelta(minutes=10,seconds=3)
        w = duration
        s_up = n+lag
        e_up = g[s_up:s_up+w][(g[s_up:s_up+w].co2_dry.diff().abs() > limit) == True].first_valid_index() - jit
        d_up = (e_up-s_up)
        if e_up is pd.NaT:
            e_up = s_up + w - jit
            d_up = (e_up-s_up)
        elif  (jit*15 < d_up < w*0.98):
            e_up -= jit*15
            d_up = (e_up-s_up)
        s_dn = s_up+up2dn
        e_dn = g[s_dn:s_dn+w][(g[s_dn:s_dn+w].co2_dry.diff().abs() > limit) == True].first_valid_index() - jit
        d_dn = (e_dn-s_dn)
        if e_dn is pd.NaT:
            e_dn = s_dn + w - jit
            d_dn = (e_dn-s_dn)
        elif (jit*15 < d_dn < w*0.98):
            e_dn -= jit*15
            d_dn = (e_dn-s_dn)
        ret.append({'timestamp':n
        ,'updraft':g[s_up:e_up].co2_dry.mean(),'updraft_max':g[s_up:e_up].co2_dry.max(),'updraft_min':g[s_up:e_up].co2_dry.min(),'updraft_duration[sec]': d_up/pd.Timedelta(seconds=1)
        ,'downdraft':g[s_dn:e_dn].co2_dry.mean(),'downdraft_max':g[s_dn:e_dn].co2_dry.max(),'downdraft_min':g[s_dn:e_dn].co2_dry.min(),'downdraft_duration[sec]':d_dn/pd.Timedelta(seconds=1)
        ,'updraft_h2o':g[s_up:e_up][h2olabel].mean(),'downdraft_h2o':g[s_dn:e_dn][h2olabel].mean()
        })
        fig = None
        try:
            if co2label == 'CO2digi2':
                sub = '_2'
                ylim = (300,600)
            elif co2label.startswith('X12CO2'):
                sub = '_3'
                ylim = (300,600)
            elif co2label.startswith('X13CO2'):
                sub = '_4'
                ylim = (3,6)
            else:
                sub = '_1'
                ylim = (300,600)
            fig_fn = pathlib.Path(f'charts/{n.strftime("%Y_%m%d_%H%M")}{sub}.png')
#            if ((d_up < w*0.98) or (d_dn < w*0.98)) and not fig_fn.is_file():
            if not fig_fn.is_file():
                mpl.use('Agg')
                pd.plotting.register_matplotlib_converters()
                fig, ax = plt.subplots(figsize=(6,4),dpi=200)
                ax.set_ylim(*ylim)
                ax.set_xlim(n.to_datetime64(),(n+pd.Timedelta(minutes=30)).to_datetime64())
                ax.axhline(ret[-1]['updraft'],color='red',linewidth=0.5)
                ax.axhline(ret[-1]['downdraft'],color='blue',linewidth=0.5)
                ax.axvspan(s_up,e_up, facecolor='tab:orange', alpha=0.5)
                ax.axvspan(s_dn,e_dn, facecolor='tab:green', alpha=0.5)
                g['co2_dry'].plot(ax=ax)
                if 'CO2digi2' in g.columns:
                    g['CO2digi'].plot(color='grey',ax=ax)
                    g['CO2digi2'].plot(color='green',ax=ax)
                ax.legend()
                ax.set_title(fig_fn,y=0.9)
                fig.tight_layout()
                fig.savefig(fig_fn)
        except Exception as e:
            print(type(g),type(g.index))
            print(fig_fn,e,s_up,e_up,s_dn,e_dn)
            raise
        finally:
            if fig is not None:
                fig.clf()
                plt.close(fig)
    return ret
def main():
    # files = sorted(pathlib.Path('../../collect/ECdata').glob('TOB3_*ECdata*.zip'),key=lambda x: x.name)
    files = sorted(pathlib.Path('../tower').glob('TOB1_*ECdata*.dat'),key=lambda x: x.name)
    files = list(list(g) for k,g in itertools.groupby(files,files_grouper))
    # files_accum = sorted(pathlib.Path('../../collect/EA_down').glob('TOB3_*EA_down*.zip'),key=lambda x: x.name)
    files_accum = sorted(pathlib.Path('../house').glob('TOB1_*EA_down*.dat'),key=lambda x: x.name)
    files_accum = list(list(g) for k,g in itertools.groupby(files_accum,files_grouper))
    files_pic = [[ff] for ff in sorted(pathlib.Path('../pic13C').glob('pic_*.zip'),key=lambda x: x.name)]
    option = {'start':pd.Timestamp('2021-04-01T17:00',tz="Asia/Tokyo"),'co2label':'CO2digi','h2olabel':'H2Odigi'}
    # start = pd.Timestamp('2021-04-28T11:30',tz="Asia/Tokyo") if option['co2label'].endswith('2') else pd.Timestamp('2020-08-12T17:00',tz="Asia/Tokyo")
    # start = pd.Timestamp('2021-04-28T11:30',tz="Asia/Tokyo")
    # option.update({'start':start})
    # files = ['ECmean.pkl']
    # eddy = pd.concat([cal_eddy(f) for f in files]).sort_index()
    with ThreadPoolExecutor() as worker:
        ret_eddy = worker.map(cal_eddy,files)
        eddy = pd.concat(ret_eddy).sort_index()
    with ProcessPoolExecutor() as worker:
        dfs = worker.map(partial(cal_conc,**option),files_accum)
        ret_accum = (d for r in filter(None,dfs) for d in r)
        dfaccum = pd.DataFrame(ret_accum).set_index('timestamp').sort_index()
        option.update({'start':pd.Timestamp('2021-04-28T11:30',tz="Asia/Tokyo"),'co2label':'CO2digi2','h2olabel':'H2Odigi2'})
        dfs2 = worker.map(partial(cal_conc,**option),files_accum)
        ret_accum2 = (d for r in filter(None,dfs2) for d in r)
        dfaccum2 = pd.DataFrame(ret_accum2).set_index('timestamp').sort_index()

        option.update({'start':pd.Timestamp('2021-10-15T22:00',tz="Asia/Tokyo"),'co2label':'X12CO2_dry','h2olabel':'H2O','filetype':'pic'})
        dfs3 = worker.map(partial(cal_conc,**option),files_pic)
        ret_accum3 = (d for r in filter(None,dfs3) for d in r)
        dfaccum3 = pd.DataFrame(ret_accum3).set_index('timestamp').sort_index()
        option.update({'start':pd.Timestamp('2021-10-15T22:00',tz="Asia/Tokyo"),'co2label':'X13CO2_dry','h2olabel':'H2O','filetype':'pic'})
        dfs4 = worker.map(partial(cal_conc,**option),files_pic)
        ret_accum4 = (d for r in filter(None,dfs4) for d in r)
        dfaccum4 = pd.DataFrame(ret_accum4).set_index('timestamp').sort_index()

    tea_w_abs = 'aw_abs'
    df = dfaccum.join(dfaccum2.rename(columns=lambda x: x.replace('draft','draft2'))).join(eddy)
    df = df.join(dfaccum3.rename(columns=lambda x: x.replace('draft','draft12c')))
    df = df.join(dfaccum4.rename(columns=lambda x: x.replace('draft','draft13c')))
    print(df[df.index.duplicated(keep='last')].index)
    df = df[~df.index.duplicated(keep='last')].asfreq('30Min').sort_index()
    mVoffset = pd.Series(83.0,index=df.index).where(pd.Timestamp('2021-01-07T12:00',tz='Asia/Tokyo') < df.index,50.0)# offset change due to wiring
    mVoffset = mVoffset.where(pd.Timestamp('2021-07-20T12:00',tz='Asia/Tokyo') > df.index, 0.0) # new SAT550 isolated from EC measurement
    df = df.assign(Ts_org=df.Ts,Ts=df.Ts-0.1*mVoffset)
    checkmode00 = pd.concat([
        # df.loc["2021-12-09T00:00":"2022-04-23T00:00",['updraft','downdraft']].between_time('01:30','01:30'),
        # df.loc["2022-04-23T00:00":,['updraft','downdraft']].between_time('01:30','02:00')
        df.loc["2021-12-09T00:00":"2022-04-23T00:00",['updraft','downdraft']].at_time('01:30:00'),
        df.loc["2022-04-23T00:00":,['updraft','downdraft']].at_time('01:30:00')
    ]).sort_index()
    checkmode00 = checkmode00[~checkmode00.index.duplicated(keep='last')]
    checkmode30 = pd.concat([
        # df.loc["2021-12-09T00:00":"2022-04-23T00:00",['updraft','downdraft']].shift(freq='30min').between_time('02:00','02:00'),
        # df.loc["2022-04-23T00:00":,['updraft','downdraft']].between_time('01:30','02:00')
        df.loc["2021-12-09T00:00":"2022-04-23T00:00",['updraft','downdraft']].shift(freq='30min').at_time('02:00:00'),
        df.loc["2022-04-23T00:00":,['updraft','downdraft']].at_time('02:00:00')
    ]).sort_index()
    checkmode30 = checkmode30[~checkmode30.index.duplicated(keep='last')]

    chk00 = (checkmode00.updraft-checkmode00.downdraft).asfreq('60Min').interpolate('time')
    chk30 = (checkmode30.updraft-checkmode30.downdraft).asfreq('60Min').interpolate('time')
    df = df.assign(check_diff=chk00.asfreq('30Min').combine_first(chk30))

    pa = pd.read_parquet('../press.parquet')
    df = df.assign(pa=pa.asfreq('30Min').pa.interpolate()*100.0)
    vm = 8.314*(1.0/(1.0+0.32*0.0000)*(df.Ts+273.15))/df.pa.fillna(value=100_000.0)
    vc = (df.updraft*df.flow_updraft+df.downdraft*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    vc_check = ((df.updraft-df.check_diff)*df.flow_updraft+df.downdraft*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    fc = 0.5*df[tea_w_abs]*(df.updraft - df.downdraft)/vm
    fc_cor = fc + df.w*(0.5*(df.updraft+df.downdraft)-vc)/vm
    fc_cor_check = 0.5*df[tea_w_abs]*(df.updraft - df.downdraft - df.check_diff)/vm + df.w*(0.5*(df.updraft+df.downdraft-df.check_diff)-vc_check)/vm

    vc2 = (df.updraft2*df.flow_updraft+df.downdraft2*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    fc2 = 0.5*df[tea_w_abs]*(df.updraft2 - df.downdraft2)/vm
    fc2_cor = fc2 + df.w*(0.5*(df.updraft2+df.downdraft2)-vc2)/vm

    vc12c = (df.updraft12c*df.flow_updraft+df.downdraft12c*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    fc12c = 0.5*df[tea_w_abs]*(df.updraft12c - df.downdraft12c)/vm
    fc12c_cor = fc12c + df.w*(0.5*(df.updraft12c+df.downdraft12c)-vc12c)/vm
    vc13c = (df.updraft13c*df.flow_updraft+df.downdraft13c*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    fc13c = 0.5*df[tea_w_abs]*(df.updraft13c - df.downdraft13c)/vm
    fc13c_cor = fc13c + df.w*(0.5*(df.updraft13c+df.downdraft13c)-vc13c)/vm
    rVPDB = 0.0111802
    k_cept_flux=(fc13c_cor/fc12c_cor/rVPDB-1)*1000.0

    renamer = {
    'updraft':'updraft_co2_dry[ppm]',
    'downdraft':'downdraft_co2_dry[ppm]',
    'updraft2':'updraft2_co2_dry[ppm]',
    'downdraft2':'downdraft2_co2_dry[ppm]',
    'flow_updraft':'flow_updraft[l 30min-1]',
    'flow_downdraft':'flow_downdraft[l 30min-1]',
    'setting_updraft':'setting_updraft[l 30min-1]',
    'setting_downdraft':'setting_downdraft[l 30min-1]',
    'setting_updraft_cor':'setting_updraft_cor[l 30min-1]',
    'setting_downdraft_cor':'setting_downdraft_cor[l 30min-1]',
    'up_count_over':'up_count_over[count 30min-1]',
    'down_count_over':'down_count_over[count 30min-1]',
    'u':'u[m sec-1]',
    'w':'w[m sec-1]',
    'w_abs':'w_abs[m sec-1]',
    'au':'au[m sec-1]',
    'aw':'aw[m sec-1]',
    'aw_abs':'aw_abs[m sec-1]',
    'Ts':'Ts[degC]',
    'Ts_org':'Ts_org[degC]',
    'ustar2':'ustar2[m2 sec-2]',
    'Ux_spike':'Ux_spike[count 30min-1]',
    'Uy_spike':'Uy_spike[count 30min-1]',
    'Uz_spike':'Uz_spike[count 30min-1]',
    'tea_flux':'tea_flux[umol m-2 sec-1]',
    'tea_flux_cor':'tea_flux_cor[umol m-2 sec-1]',
    'tea_flux_cor_checkmode':'tea_flux_cor_checkmode[umol m-2 sec-1]',
    'vc':'vc[ppm]',
    'vc_check':'vc_check[ppm]',
    'tea_flux2':'tea_flux2[umol m-2 sec-1]',
    'tea_flux2_cor':'tea_flux2_cor[umol m-2 sec-1]',
    'vc2':'vc2[ppm]',
    'k_cept':'(13C/12C/rPVDB-1)*1000',
    'check_diff':'check_diff[ppm]',
    }
    renamer |= {c:f'{c}[mmol mol-1]' for c in df.columns if c.endswith('_h2o') and not c.endswith('c_h2o')}
    df = df.assign(tea_flux=fc,tea_flux_cor=fc_cor,tea_flux_cor_checkmode=fc_cor_check,vc=vc,vc_check=vc_check,tea_flux2=fc2,tea_flux2_cor=fc2_cor,vc2=vc2)
    df = df.assign(tea_flux12c=fc12c,tea_flux12c_cor=fc12c_cor,vc12c=vc12c)
    df = df.assign(tea_flux13c=fc13c,tea_flux13c_cor=fc13c_cor,vc13c=vc13c)
    df = df.assign(k_cept=k_cept_flux)
    df = df.groupby(level=0).last()
    cflux = [c for c in df.columns if c.startswith('tea_flux')]
    errors = [
        (slice('2021-12-02T10:30','2021-12-02T10:30'),cflux,float('nan'))
        ,(slice('2021-12-02T15:00','2021-12-02T16:00'),cflux,float('nan'))
    ]
    for r,c,v in errors:
        df.loc[r,c] = v
    df['2021-08-22':].rename(columns=renamer).to_csv('KEW_tea_mean4.csv',date_format='%Y-%m-%d %H:%M')
    df['2021-08-22':].resample('1H').first().rename(columns=renamer).to_csv('KEW_tea_mean4_0000.csv',date_format='%Y-%m-%d %H:%M')
    df['2021-08-22':].resample('1H',offset='30Min').first().rename(columns=renamer).to_csv('KEW_tea_mean4_0030.csv',date_format='%Y-%m-%d %H:%M')
    dfcheck = df.loc["2021-12-09T00:00":,:].between_time('01:30','01:30')
    dfcheck = pd.concat([
        df.loc["2021-12-09T00:00":"2022-04-23T00:00",:].between_time('01:30','01:30'),
        df.loc["2022-04-23T00:00":,:].between_time('01:30','02:00')
    ]).sort_index()
    dfcheck = dfcheck[~dfcheck.index.duplicated(keep='last')]
    dfcheck.assign(co2_dry_diff=dfcheck.updraft-dfcheck.downdraft).rename(columns=renamer).to_csv('KEW_tea_check4.csv',date_format='%Y-%m-%d %H:%M')

    # ec = pd.read_excel('211220_TEA_KEW.xlsx',usecols=[0,1,2,3],index_col=[0])
    # ec.index = ec.index.tz_localize('Asia/Tokyo').round('5S').set_names('timestamp')
    # ec.join(df['2021-08-22':].rename(columns=renamer)).to_csv('KEW_tea_mean3.csv',date_format='%Y-%m-%d %H:%M')


if __name__ == '__main__':
    main()