import pathlib
import zipfile
import tempfile
import numpy as np
import pandas as pd
class TOB1:
    datatype = {
      'IEEE4':  {'type':'<f','size':4},
      'IEEE4B': {'type':'>f','size':4},
      'IEEE8':  {'type':'<d','size':8},
      'FP2':    {'type':'>h','size':2},
      'INT4':   {'type':'<i4','size':4},
      'UINT2':  {'type':'<u2','size':2},
      'USHORT': {'type':'<H','size':2},
      'UINT4':  {'type':'<u4','size':4},
      'ULONG':  {'type':'<L','size':4},
      'LONG':   {'type':'<l','size':4},
      'SecNano':{'type':'<Q','size':8},
      'BOOL':   {'type':'<?','size':1},
      'ASCII':  {'type':'<S','size':1}
    }
    def transform_crdata(self,d,t):
        if t == np.dtype('>i2'):
            return (0x1FFF & d)*(0.1**((0x6000 & d) >> 13))*(-1)**((0x8000 & d) >> 15)#FP2 format
        elif t == np.dtype('>f4'):
            return d.byteswap().newbyteorder()
        else:
            return d
    def read_file(self,fn):
        csepoch = pd.Timestamp('1990-01-01T00:00')
        clabels = ['name','unit','method','type']
        with open(fn,'rb') as f:
            buf = f.read(8192)
            ftype = buf[0:8].decode('ascii').split(',')[0].strip('"')
            headersize = 6 if ftype == 'TOB3' else 5 if ftype == 'TOB1' or ftype == 'TOA5' else 1
            hmax = len(buf) if len(buf) < 8192 else 8192
            header = buf[0:hmax]
            headers = [[ee.strip(' "') for ee in e.decode('ascii').split(',')] for i,e in enumerate(header.split(b'\x0d\x0a')) if i < headersize]
            self.header_offset = 0
            self.headers = headers
            for i in range(headersize):
                self.header_offset = header.find(b'\x0d\x0a',self.header_offset)+2
            if ftype in ('TIMESTA','TOA5'):
                f.seek(0)
            else:
                for h in headers[5 if ftype == 'TOB3' else 4]:
                    if h.startswith('ASCII'):
                        lidx,ridx = h.find('('),h.find(')')
                        if lidx != -1 and ridx != -1:
                            size = int(h[lidx+1:ridx])
                            self.datatype[h] = {'type':f'<S{size}','size':size}
                if ftype == 'TOB3':
                    val_stamp = int(headers[1][4])
                    framesize = int(headers[1][2])
                    data_length = len(headers[5])
                    data_size = sum(self.datatype[h]['size'] for h in headers[5])
                    frame_line_number = (framesize-16)//data_size
                    # frameformat = [('time_offset_','int32'),('subseconds_','int32'),('beg_','int32')]
                    # frameformat.extend([(name, self.datatype[h]['type']) for name,h in zip(headers[2],headers[5])])
                    # frameformat.append(('validation_','int32'))
                    frameformat = [('time_offset_','int32'),('subseconds_','int32'),('beg_','int32'),('data_','b',framesize-16),('validation_','int32')]
                    dataformat = [(name, self.datatype[h]['type']) for name,h in zip(headers[2],headers[5])]
                    frame_td_sec = pd.Timedelta(headers[1][1].lower().replace('sec','',1).replace('usec',' microsecond').replace('msec','millisecond'))/pd.Timedelta(seconds=1)
                    sub_td = pd.Timedelta(headers[1][5].lower().replace('sec','',1).replace('usec',' microsecond').replace('msec','millisecond'))/pd.Timedelta(milliseconds=1)
                else:
                    frameformat = []
                    framesize   = 0
                    for fi,t in zip(headers[1],headers[4]):
                        frameformat.append((fi,'{type}'.format(**self.datatype[t])))
                        framesize   += self.datatype[t]['size']
        try:
            if ftype not in ('TIMESTA','TOA5'):
                dt = np.dtype(frameformat)
                all_length = None
                mm = np.memmap(fn,dtype=dt,mode='r',offset=self.header_offset)
                all_length = len(mm)
                start = self.offset
                end = self.offset+self.length if (self.offset+self.length) < all_length else all_length
                # end = -(all_length % framesize) if (all_length % framesize) > 0 else None
                chunk = mm[start:end]
                if ftype == 'TOB3':
                    chunk = chunk[((0xFFFF0000 & chunk['validation_']) >> 16) == val_stamp]
                    chunk = chunk[(0x000007FF & chunk['validation_']) == 0]
                    time_  = csepoch + pd.to_timedelta(np.vstack([chunk['time_offset_']+i*frame_td_sec for i in range(frame_line_number)]).flatten('F'),unit='s')
                    time_ += pd.to_timedelta(sub_td*np.vstack([chunk['subseconds_'] for i in range(frame_line_number)]).flatten('F'),unit='ms')
                    beg_ = np.vstack([chunk['beg_']+i for i in range(frame_line_number)]).flatten('F')
                    chunk = np.frombuffer(chunk['data_'].tobytes(),dtype=np.dtype(dataformat))
                    excols = ['time_offset_','subseconds_','beg_','validation_']
                else:
                    time_ = csepoch + pd.to_timedelta(chunk['SECONDS'],unit='s') + pd.to_timedelta(chunk['NANOSECONDS'],unit='ns')
                    beg_ = chunk['RECORD'] if 'RECORD' in chunk.dtype.fields.keys() else np.arange(len(time_))
                    excols = ['SECONDS','NANOSECONDS','RECORD']
                colnames = np.array([(c,t[0]) for c,t in chunk.dtype.fields.items() if c not in excols and t[0].shape == ()])
                newdtype = [(c,str(t).replace('>i2','<f2').replace('>f4','<f4')) for c,t in colnames]
                dd = np.rec.array([self.transform_crdata(chunk[c],t) for c,t in colnames],dtype=newdtype)
            if ftype == 'TIMESTA':
                df = pd.read_csv(fn)
                df.index = pd.DatetimeIndex(pd.to_datetime(df.TIMESTAMP_END,format='%Y%m%d%H%M'))
                RECORD = None
                frameformat = None
                framesize = None
            elif ftype == 'TOA5':
                df = pd.read_csv(fn,skiprows=[0,2,3],parse_dates=['TIMESTAMP'],index_col='TIMESTAMP',na_values=['NAN'])
                RECORD = pd.Series(df.RECORD.astype('uint32'),index=df.index,name='idx_RECORD')
                frameformat = None
                framesize = None
                print(df.head())
            else:
                df = pd.DataFrame.from_records(dd,columns=colnames[:,0],index=time_,coerce_float=True)
                RECORD = pd.Series(beg_,index=df.index,name='idx_RECORD').astype('uint32')
            for c in df.select_dtypes([np.object]).columns:
                df[c] = df[c].str.decode('utf-8').replace(r'\x00.*','',regex=True)
            SECONDS     = pd.Series((df.index - csepoch)//pd.Timedelta(seconds=1),index=df.index,name='idx_SECONDS').astype('uint32')
            NANOSECONDS = pd.Series((df.index - df.index.floor('1S'))//pd.Timedelta(nanoseconds=1),index=df.index,name='idx_NANOSECONDS').astype('uint32')
            if any(c in df.columns for c in ('SECONDS','NANOSECONDS','RECORD')):
                df = df.drop(columns=[c for c in ('SECONDS','NANOSECONDS','RECORD') if c in df.columns])
            df = pd.concat([SECONDS,NANOSECONDS,RECORD,df],axis=1)
            self.df = df
#            self.mcolumns = pd.MultiIndex.from_tuples(list(zip(*headers[1:])),names=clabels)
#            if multicolumns:
#                self.df.columns = self.mcolumns
            self.df.index.name = 'timestamp'
            self.df.index = self.df.index.tz_localize(self.tz)
            self.frameformat = frameformat
            self.framesize = framesize
            self.headers = headers
        except Exception as e:
            print(e)
            print(f'can not read {fn}')
            print(headers[1])
            print(f'framesize={framesize}')
            print(dt)
            print(self.header_offset)
            if all_length:
                print(all_length%framesize)
            self.df = None
    def __init__(self,fn,tz='Asia/Tokyo',multicolumns=False,offset=0,length=8640000,record=False):
        fn = pathlib.Path(fn)
        self.offset = offset
        self.length = length
        self.tz = tz
        if fn.suffix.lower()== '.zip':
            with tempfile.TemporaryDirectory() as tmp,zipfile.ZipFile(fn,'r') as myzip:
                myzip.extract(fn.with_suffix('.dat').name,tmp)
                fn = pathlib.Path(tmp)/fn.with_suffix('.dat').name
                self.read_file(fn)
        else:
            self.read_file(fn)
