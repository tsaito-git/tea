#coding: utf8
import itertools
import pathlib
import pandas as pd
import tob

#TOB1_basefilename_yyyy_mm_dd_hhmn_n.DAT
EA_folder = pathlib.Path("D:/TEA/KEWdata/tob1")
EA_calculated = pathlib.Path('eddy')
if not EA_calculated.is_dir():
    EA_calculated.mkdir()
files_grouper = lambda x: x.name[-16:]
files = sorted(EA_folder.glob('TOB1_*ECdata*.dat'),key=files_grouper)
files = list(list(g) for k,g in itertools.groupby(files,files_grouper))
sum_slm = lambda x: (x/600).sum()
count_over = lambda x: (x >= 1.0 ).sum()

def cal_eddy(f):
    pkl_fn = EA_calculated/f[0].with_suffix('.pkl').name
    if pkl_fn.is_file():
        return pd.read_pickle(pkl_fn)
    dfs = []
    for ff in f:
        tmp = tob.TOB1(ff).df.copy()
        if len(tmp) > 0 and 'w' in tmp.columns:
            tmp = tmp.rename(columns={'flow_upstream':'flow_updraft','flow_downstream':'flow_downdraft'})
            dfs.append(tmp)
    df = pd.concat(dfs)
    if len(df) > 0 and 'w' in df.columns:
#        print(','.join(ff.name for ff in f))
        df = df.assign(w_abs=df.w.abs())
        gpr = pd.Grouper(freq='30min',closed='right',label='right')
        aggr = {
            'flow_updraft':('flow_updraft',sum_slm),
            'up_count_over':('flow_updraft',count_over),
            'flow_downdraft':('flow_downdraft',sum_slm),
            'down_count_over':('flow_downdraft',count_over),
            'u':('u','mean'),'w':('w','mean'),'w_abs':('w_abs','mean'),'Ts':('Ts','mean')
        }
        if 'setting_point_updraft' in df.columns:
            df = df.assign(diff_flow_updraft=df.setting_point_updraft - df.flow_updraft)
            aggr.update({'setting_updraft':('setting_point_updraft',sum_slm)})
            aggr.update({'diff_flow_updraft':('diff_flow_updraft',sum_slm)})
        if 'setting_point_downdraft' in df.columns:
            df = df.assign(diff_flow_downdraft=df.setting_point_downdraft - df.flow_downdraft)
            aggr.update({'setting_downdraft':('setting_point_downdraft',sum_slm)})
            aggr.update({'diff_flow_downdraft':('diff_flow_downdraft',sum_slm)})
        r = df.groupby(gpr)
        dfavg = r.aggregate(**aggr)
        dfedy = df[['u','w']].groupby(gpr).transform(lambda x: x-x.mean())
        ustar2 = dfedy.groupby(gpr).apply(lambda r: -((r.w*r.u).mean()) )
        dfavg = dfavg.assign(ustar2=ustar2)
        dfavg.to_pickle(pkl_fn)
        return dfavg
ret_eddy = map(cal_eddy,files)
eddy = pd.concat(ret_eddy).sort_index()
eddy = eddy[~eddy.index.duplicated(keep='last')].asfreq('30min')
eddy.to_csv('EAstat.csv',date_format='%Y-%m-%d %H:%M')
