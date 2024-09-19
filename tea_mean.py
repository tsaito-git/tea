#coding: utf8
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from logging import getLogger
import math
import pathlib
import itertools

import pandas as pd

from tob import TOB1 as TOB

logger = getLogger(__name__)

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
    return x.where(limit_in,float('nan')).ffill(),(~limit_in).sum()#spikes(>3.5*std) padded,spike count
def cal_eddy(f,timezone=None):
    #cache
    fld_eddy = pathlib.Path('eddy')
    if not fld_eddy.is_dir():
        fld_eddy.mkdir(parents=True)
    pkl_fn = fld_eddy/f[0].with_suffix('.parquet').name
    if pkl_fn.is_file():
        return pd.read_parquet(pkl_fn)
    #calculate statistics
    dfs = []
    df = None
    for ff in f:
        tmp = TOB(ff,tz=timezone).df.copy()
        if len(tmp) > 0 and 'w' in tmp.columns:
            tmp = tmp.rename(columns={'flow_upstream':'flow_updraft','flow_downstream':'flow_downdraft'})
            dfs.append(tmp)
    if len(dfs) > 0:
        df = pd.concat(dfs)
    if df is not None and len(df) > 0 and 'w' in df.columns:
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
        dfedy_dr = r.apply(double_rotation)
        dfavg = dfavg.join(dfedy_dr)
        dfavg.to_parquet(pkl_fn)
        return dfavg
def cal_conc(f,co2label,h2olabel,start,timezone):
    #load concentration data
    tob1 = pd.concat([TOB(ff,tz=timezone).df for ff in f])
    df = tob1.sort_index().query('@start <= index')
    if len(df) == 0 or (co2label not in df.columns):
        return None
    df = df.assign(co2_dry=df[co2label]/(1-df[h2olabel]/1000.0))
    #estimate concentration of two bags
    baselag = pd.Timedelta(seconds=50)
    limit = 4.0
    ret = []
    jit = pd.Timedelta(seconds=1)
    for n,g in df.groupby(pd.Grouper(freq='30Min',closed='right',label='left')):
        lag = baselag
        duration = pd.Timedelta(minutes=9,seconds=0)
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
    return ret
def main():

    option = {'start':pd.Timestamp('2022-05-01T00:00',tz="Asia/Tokyo"),'co2label':'CO2digi','h2olabel':'H2Odigi','timezone':'Asia/Tokyo'}
    
    fld_ECdata = '../../tower'
    fld_EAdown = '../../house'
    PAdata = '../../press.parquet'

    files = sorted(pathlib.Path(fld_ECdata).glob('TOB1_*ECdata*.dat'),key=files_grouper)
    files = filter(lambda x: x.stem > 'TOB1_{:%Y_%m_%d}'.format(option['start']),files)
    files = list(list(g) for k,g in itertools.groupby(files,files_grouper))
    files_accum = sorted(pathlib.Path(fld_EAdown).glob('TOB1_*EA_down*.dat'),key=files_grouper)
    files_accum = filter(lambda x: x.stem > 'TOB1_{:%Y_%m_%d}'.format(option['start']),files_accum)
    files_accum = list(list(g) for k,g in itertools.groupby(files_accum,files_grouper))
 
    #calculate eddy statistics
    with ThreadPoolExecutor() as worker:
        ret_eddy = worker.map(partial(cal_eddy,timezone=option['timezone']),files)
        eddy = pd.concat(ret_eddy).sort_index()
    #estimate concentration of bags
    with ThreadPoolExecutor() as worker:
        dfs = worker.map(partial(cal_conc,**option),files_accum)
        ret_accum = (d for r in filter(None,dfs) for d in r)
        dfaccum = pd.DataFrame(ret_accum).set_index('timestamp').sort_index()

    tea_w_abs = 'aw_abs'
    df = dfaccum.join(eddy)
    df = df[~df.index.duplicated(keep='last')].asfreq('30Min').sort_index()

    mVoffset = 0.0
    df = df.assign(Ts_org=df.Ts,Ts=df.Ts.astype('float64')-0.1*mVoffset)

    #for checkmode
    checkmode00 = df[['updraft','downdraft']].at_time('01:30:00').sort_index()
    checkmode00 = checkmode00[~checkmode00.index.duplicated(keep='last')]
    checkmode30 = df[['updraft','downdraft']].at_time('02:00:00').sort_index()
    checkmode30 = checkmode30[~checkmode30.index.duplicated(keep='last')]

    chk00 = (checkmode00.updraft-checkmode00.downdraft).asfreq('60Min').interpolate('time')
    chk30 = (checkmode30.updraft-checkmode30.downdraft).asfreq('60Min').interpolate('time')
    df = df.assign(check_diff=chk00.asfreq('30Min').combine_first(chk30))

    #load air pressure data
    pa = pd.read_parquet(PAdata)# air pressure(hPa)
    df = df.assign(pa=pa.asfreq('30Min').pa.interpolate()*100.0)

    #calculate TEA flux
    vm = 8.314*(df.Ts+273.15)/df.pa.fillna(value=100_000.0)# no correction for Ts
    vc = (df.updraft*df.flow_updraft+df.downdraft*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    vc_check = ((df.updraft-df.check_diff)*df.flow_updraft+df.downdraft*df.flow_downdraft)/(df.flow_updraft+df.flow_downdraft)
    fc = 0.5*df[tea_w_abs]*(df.updraft - df.downdraft)/vm
    fc_cor = fc + df.w*(0.5*(df.updraft+df.downdraft)-vc)/vm
    fc_cor_check = 0.5*df[tea_w_abs]*(df.updraft - df.downdraft - df.check_diff)/vm + df.w*(0.5*(df.updraft + df.downdraft - df.check_diff) - vc_check)/vm

    renamer = {
        'updraft':'updraft_co2_dry[ppm]',
        'downdraft':'downdraft_co2_dry[ppm]',
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
        'check_diff':'check_diff[ppm]',
    }
    renamer |= {c:f'{c}[mmol mol-1]' for c in df.columns if c.endswith('_h2o') and not c.endswith('c_h2o')}
    df = df.assign(tea_flux=fc,tea_flux_cor=fc_cor,tea_flux_cor_checkmode=fc_cor_check,vc=vc,vc_check=vc_check)
    df = df.groupby(level=0).last()
    df.rename(columns=renamer).to_csv('KEW_tea_mean.csv',date_format='%Y-%m-%d %H:%M')

    dfcheck = df.between_time('01:30','02:00').sort_index()
    dfcheck = dfcheck[~dfcheck.index.duplicated(keep='last')]
    dfcheck.assign(co2_dry_diff=dfcheck.updraft-dfcheck.downdraft).rename(columns=renamer).to_csv('KEW_tea_check.csv',date_format='%Y-%m-%d %H:%M')

if __name__ == '__main__':
    main()