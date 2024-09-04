import datetime
import pathlib
from struct import Struct
from collections import namedtuple
from zipfile import ZipFile
from contextlib import contextmanager,ExitStack
class TOB:
    """
    Campbell Scientific TOB1/3 and TOA5 format reader
    """
    tdunits = {'USEC':'microseconds','MSEC':'milliseconds','SEC':'seconds','MIN':'minutes','HOUR':'hours'}
    sub_tds = {'Sec100Usec':datetime.timedelta(microseconds=100),'SecMsec':datetime.timedelta(milliseconds=1)}
    datatype = {
        'IEEE4':  {'type':'<f','size':4},
        'IEEE4B': {'type':'>f','size':4},
        'IEEE8':  {'type':'<d','size':8},
        'FP2':    {'type':'>h','size':2},
        'INT4':   {'type':'>l','size':4},
        'UINT2':  {'type':'>H','size':2},
        'USHORT': {'type':'<H','size':2},
        'UINT4':  {'type':'>L','size':4},
        'ULONG':  {'type':'<L','size':4},
        'LONG':   {'type':'<l','size':4},
        'SecNano':{'type':'<Q','size':8},
        'BOOL':   {'type':'<?','size':1},
        'ASCII':  {'type':'<s','size':1}
    }
    dtype_type = namedtuple('dtype',['name','type','conv','offset','size'])
    record_name_ = 'recnbr'
    timestamp_name_ = 'timestamp'
    csepoch = datetime.datetime(1990,1,1)
    @classmethod
    def header_convert(cls,headers3):
        if headers3[0][0] == 'TOB3':
            headers1 = [
                ['TOB1']+headers3[0][1:7]+[headers3[1][0]],
                ['SECONDS','NANOSECONDS','RECORDS']+headers3[2],
                ['SECONDS','NANOSECONDS','RN']+headers3[3],
                ['','','']+headers3[4],
                ['ULONG','ULONG','ULONG']+headers3[5]
            ]
            idx_cols = 2
        elif headers3[0][0] == 'TOB1':
            idx_cols = 1
            headers1 = headers3
        elif headers3[0][0] == 'TOA5':
            idx_cols = 1
            headers3.append(['']*len(headers3[idx_cols]))
            headers3[0][0] = 'TOB1'
            headers1 = headers3
        else:
            return
        units = {c:u for c,u in zip(headers3[idx_cols],headers3[idx_cols+1])}
        methods = {c:m for c,m in zip(headers3[idx_cols],headers3[idx_cols+2])}
        types = {c:t for c,t in zip(headers3[idx_cols],headers3[idx_cols+3])}
        return headers1,units,methods,types
    @classmethod
    def converter_from(cls,sfmt,offset):
        unpacker = Struct(sfmt)
        def conv(buf):
            dd, = unpacker.unpack_from(buf,offset)
            return dd
        return conv
    @classmethod
    def fp2_converter_from(cls,sfmt,offset):
        unpacker = Struct(sfmt)
        def conv(buf):
            dd, = unpacker.unpack_from(buf,offset)
            return (0x1FFF & dd)*(0.1**((0x6000 & dd) >> 13))*(-1)**((0x8000 & dd) >> 15)
        return conv
    @classmethod
    def text_converter_from(cls,sfmt,offset):
        unpacker = Struct(sfmt)
        def conv(buf):
            dd, = unpacker.unpack_from(buf,offset)
            return dd.split(b'\x00',1)[0].decode('utf-8')
        return conv
    @classmethod
    def data_converter(cls,strdata):
        try:
            return float(strdata)
        except:
            return strdata
    @classmethod
    def convert_dtype(cls,headers,removesuffix=[],rename=[]):
        start_bytes = 0
        ftype = headers[0][0]
        ret = []
        if ftype == 'TOB3':
            columns = headers[2]
            types = headers[5]
        elif ftype == 'TOB1':
            data_start = 3 if headers[1][2] == 'RECORD' else 2
            columns = headers[1][data_start:]
            types = headers[4][data_start:]
        elif ftype == 'TOA5':
            columns = headers[1]
            types = headers[2]
        for h,t in zip(columns,types):
            h = h.replace('(','_').replace(')','_')
            for r in removesuffix:
                if h.endswith(r):
                    h = h[:-len(r)]
                    break
            for r in rename:
                h = h.replace(*r.split('=',1))
            if ftype == 'TOA5':
                size = 1
                if t == 'TS':
                    h = TOB.timestamp_name_
                    u = datetime.datetime.fromisoformat
                elif t == 'RN':
                    h = TOB.record_name_
                    u = int
                else:
                    u = cls.data_converter
                ret.append(cls.dtype_type(h,t,u,start_bytes,size))
            else:
                if t.startswith('ASCII('):
                    size = int(t[6:-1])
                    u = cls.text_converter_from('<{}s'.format(size),start_bytes)
                elif t == 'FP2':
                    size = cls.datatype[t]['size']
                    u = cls.fp2_converter_from(cls.datatype[t]['type'],start_bytes)
                else:
                    size = cls.datatype[t]['size']
                    u = cls.converter_from(cls.datatype[t]['type'],start_bytes)
                ret.append(cls.dtype_type(h,t,u,start_bytes,size))
            start_bytes += size
        return tuple(ret)
    @classmethod
    def read_header(cls,fp):
        with cls.open_file(fp) as f:
            header = f.read(8192)
            if header[1:5] == b'TOB3':
                headersize = 6
            elif header[1:5] == b'TOB1':
                headersize = 5
            elif header[1:5] == b'TOA5':
                headersize = 4
            else:
                headersize = 1
            headers = header.split(b'\x0d\x0a',headersize)
            headers = [[c.strip().strip('"') for c in h.decode('utf-8').split(',')] for h in headers[:headersize]]
            header_offset = 0
            for i in range(headersize):
                header_offset = header.find(b'\x0d\x0a',header_offset)+2
            return header_offset,headers
    @classmethod
    @contextmanager
    def open_file(cls,fp):
        with ExitStack() as stack:
            if fp.suffix.lower() == '.zip':
                mzip = stack.enter_context(ZipFile(fp))
                fn, = [fz for fz in mzip.infolist() if not fz.is_dir() and fz.filename[:-4].endswith(fp.stem)]
                f = stack.enter_context(mzip.open(fn))
            elif fp.suffix.lower() in ('.dat','.tob','.toa'):
                f = stack.enter_context(open(fp,'rb'))
            yield f
    def __init__(self,fp,removesuffix=[],rename=[],all_data=True):
        self.fp = pathlib.Path(fp) if not isinstance(fp,pathlib.Path) else fp
        self.header_offset,self.headers = TOB.read_header(self.fp)
        self.ftype = self.headers[0][0]
        self.df_ = None
        self.dtype = TOB.convert_dtype(self.headers,removesuffix,rename)
        if self.ftype == 'TOB3':
            self.dsize = sum(t.size for t in self.dtype)
            self.hfmt = Struct('<LLL')
            self.ffmt = Struct('<L')
            self.result_type = namedtuple(self.ftype,[TOB.timestamp_name_,TOB.record_name_]+[t.name for t in self.dtype])
            self.readblock = self.readblock1
            self.read_frame = self.read_frame3
            self.framesize = int(self.headers[1][2])
            td,unit = self.headers[1][1].split(' ',1)
            self.td = datetime.timedelta(**{TOB.tdunits[unit]:int(td)})
            self.sub_td = TOB.sub_tds[self.headers[1][5]]
            self.valid_stamp = int(self.headers[1][4])
            self.table = self.headers[1][0]
            self.ring = int(self.headers[1][6])
            self.lastcardtime = TOB.csepoch + datetime.timedelta(seconds=int(self.headers[1][7]))
            self.lastbeg = 0
        elif self.ftype == 'TOB1':
            self.dsize = sum(t.size for t in self.dtype)
            self.hfmt = Struct('<LLL' if self.headers[1][2] == 'RECORD' else '<LL')
            self.ffmt = Struct('<L')
            hcol = [TOB.timestamp_name_]
            if self.headers[1][2] == 'RECORD':
                hcol.append(TOB.record_name_)
            self.result_type = namedtuple(self.ftype,hcol+[t.name for t in self.dtype])
            self.readblock = self.readblock1
            self.read_frame = self.read_frame1
            self.framesize = self.hfmt.size + self.dsize
            self.sub_td = datetime.timedelta(microseconds=1)
            self.table = self.headers[0][7]
        elif self.ftype == 'TOA5':
            self.result_type = namedtuple(self.ftype,[t.name for t in self.dtype])
            self.readblock = self.readblock5
            self.table = self.headers[0][7]
        # # gpst = [Struct('{}{}'.format(k,''.join(g[0][1:]for g in gp))) for k,gp in itertools.groupby(dtype,key=lambda x: x[0][0])]
        self.columns = self.result_type._fields
        self.all_data = all_data
    def read_frame1(self,buf):            
        ts,sub_ts,*beg = self.hfmt.unpack(buf[:self.hfmt.size])
        ts = TOB.csepoch + datetime.timedelta(seconds=ts)+sub_ts//1000*self.sub_td
        start = self.hfmt.size
        yield self.result_type(ts,*beg,*map(lambda x: x.conv(buf[start:(start+self.dsize)]),self.dtype))#caution only microseconds resolution
    def read_frame3(self,buf):
        ts,sub_ts,beg = self.hfmt.unpack(buf[:self.hfmt.size])
        ts = TOB.csepoch + datetime.timedelta(seconds=ts)
        start = self.hfmt.size
        footer, = self.ffmt.unpack(buf[-self.ffmt.size:])
        foffset = (0x000007FF & footer)
        valid = (0xFFFF0000 & footer) >> 16
        num_row = (len(buf)-16-foffset)//self.dsize
        # flag_e = (0x00002000 & footer) >> 14
        # flag_m = (0x00004000 & footer) >> 15
        if (self.all_data and valid == 0) or (valid != self.valid_stamp) or (0 < beg < self.lastbeg):
            return
        self.lastbeg = beg
        for i in range(num_row):
            end = start + self.dsize
            yield self.result_type(ts+sub_ts*self.sub_td,self.lastbeg,*map(lambda x: x.conv(buf[start:end]),self.dtype))
            start = end
            self.lastbeg += 1
            ts += self.td
    def readlines(self):
        with TOB.open_file(self.fp) as f:
            self.lastbeg = 0
            yield from self.readblock(f)
    def readblock5(self,f):
        for _ in range(4): f.readline()
        for l in f:
            yield self.result_type(*map(lambda x: x[0].conv(x[1].strip('"')),zip(self.dtype,l.rstrip().split(','))))
    def readblock1(self,f):
            f.seek(self.header_offset)
            while True:
                buf = f.read(self.framesize)
                if buf == b'' or len(buf) != self.framesize:
                    break
                yield from self.read_frame(buf)
    def to_csv(self,of,append=False):
        import csv
        of = pathlib.Path(of) if not isinstance(of,pathlib.Path) else of
        with open(of,'a' if append else 'w',newline='') as f:
            csvwriter = csv.writer(f)
            if not append:
                csvwriter.writerow(self.columns)
            csvwriter.writerows(self.readlines())
    def to_sqlite3(self,of,tbname,reader=None,index_col='timestamp_ms(?1)',virtual_id_column=('id','`TimeOfRec`/1000')):
        if reader is None:
            reader = self.readlines
        import sqlite3
        sqlite_datatype = {
            'IEEE4':  'REAL',
            'IEEE4B': 'REAL',
            'IEEE8':  'REAL',
            'FP2':    'REAL',
            'INT4':   'INTEGER',
            'UINT2':  'INTEGER',
            'USHORT': 'INTEGER',
            'UINT4':  'INTEGER',
            'ULONG':  'INTEGER',
            'LONG':   'INTEGER',
            'SecNano':'INTEGER',
            'BOOL':   'INTEGER',
            'ASCII':  'TEXT'
        }
        con = sqlite3.connect(of)
        con.create_function('timestamp_ms', 1, lambda x: int(datetime.datetime.fromisoformat(x).timestamp()*1000),deterministic=True)
        try:
            create_sql = 'CREATE TABLE IF NOT EXISTS `{}` (`TimeOfRec` INTEGER PRIMARY KEY,`{}` DATETIME,`{}` INTEGER,{})'.format(tbname,TOB.timestamp_name_,TOB.record_name_,','.join('`{}` {}'.format(t.name,sqlite_datatype.get(t.type,'TEXT')) for t in self.dtype if t.type not in ('TS','RN')))
            if virtual_id_column:
                create_sql = create_sql[:-1]+',`{0}` INTEGER GENERATED ALWAYS AS ({1}) VIRTUAL)'.format(*virtual_id_column)
            con.execute(create_sql)
            if virtual_id_column:
                con.execute('CREATE UNIQUE INDEX IF NOT EXISTS `idx_id` ON `{}`(`{}`)'.format(tbname,virtual_id_column[0]))
            con.execute('CREATE UNIQUE INDEX IF NOT EXISTS `idx_timestamp` ON `{}`(`{}`)'.format(tbname,TOB.timestamp_name_))
            dbcols = {r[0] for r in con.execute('SELECT c.name FROM pragma_table_xinfo("{}") c;'.format(tbname))}
            for t in self.dtype:
                if t.name not in dbcols:
                    con.execute('ALTER TABLE `{}` ADD COLUMN `{}` {}'.format(tbname,t.name,sqlite_datatype.get(t.type,'TEXT')))
            sql = 'INSERT INTO `{}` (`TimeOfRec`,{}) VALUES ({},{}) ON CONFLICT(`TimeOfRec`)  DO UPDATE SET {}'.format(tbname,
                ','.join('`{}`'.format(c) for c in self.columns),
                index_col,
                ','.join('?{}'.format(i+1) for i,c in enumerate(self.columns)),
                ','.join('`{0}`=excluded.`{0}`'.format(c) for c in self.columns)
                )
            con.executemany(sql,reader())
            con.commit()
        finally:
            con.close()
    def __iter__(self):
        yield from self.readlines()
    def itertuples(self,index=False):
        yield from self.__iter__()
    def to_dict_gen(self,orient='records'):
        yield from map(lambda x: x._asdict(), self.__iter__())
    def to_dict(self,orient='records'):
        return list(self.to_dict_gen(orient))

try:
    from pandas import Series,Timestamp,Timedelta,concat,read_csv
    from pandas import DataFrame,DatetimeIndex,to_timedelta
    from numpy import datetime64,nditer,fromfile,datetime_as_string
    class TOBdf(TOB):
        @classmethod
        def df_tob1(cls,df,tob1file,headers=[],units={},methods={}):
            pdtype2tobtype = {
                'float32': 'IEEE4',
                'uint32': 'ULONG',
                'int32': 'LONG'
            }
            SECONDS     = Series((df.index - Timestamp(cls.csepoch,tz=df.index.tz))//Timedelta(seconds=1),index=df.index,name='idx_SECONDS').astype('uint32')
            NANOSECONDS = Series((df.index - df.index.floor('1S'))//Timedelta(nanoseconds=1),index=df.index,name='idx_NANOSECONDS').astype('uint32')
            if cls.record_name_ in df.columns:
                RECORD = df[cls.record_name_].rename('idx_RECORD').astype('uint32')
                df = df.drop(columns=[cls.record_name_])
            else:
                RECORD = Series(None,index=df.index,name='idx_RECORD').astype('uint32')
            typeconv = {c:str(t)[:-2]+'32' for c,t in zip(df.columns,df.dtypes) if str(t).endswith(('64','32'))}
            df = df[list(typeconv.keys())].astype(typeconv)
            if (len(headers) == 6 and headers[0][0] == 'TOB3') or (len(headers) == 5 and headers[0][0] == 'TOB1') or (len(headers) == 4 and headers[0][0] == 'TOA5'):#TOB1/3 headers
                headers1,units,methods,types = TOB.header_convert(headers)
                header0 = ','.join('"{}"'.format(c) for c in headers1[0])
            else:
                header0 = '"TOB1","9999","CR1000","9999","CR1000.Std.99.99","CPU:CR1000.CR1","0","Table"'
            headers = [
                header0,#header
                '"SECONDS","NANOSECONDS","RECORD",' + ','.join(f'"{c}"' for c in df.columns),#name
                '"SECONDS","NANOSECONDS","RN",' + ','.join('"{}"'.format(units.get(c,'V')) for c in df.columns),#unit
                '"","","",' + ','.join('"{}"'.format(methods.get(c,'Smp')) for c in df.columns),#method
                '"ULONG","ULONG","ULONG",' + ','.join('"{}"'.format(pdtype2tobtype.get(str(t))) for t in df.dtypes)#type
            ]
            headers = [h.encode('utf-8') for h in headers]
            with open(tob1file,'wb') as tob1:
                tob1.write(b'\x0D\x0A'.join(headers))
                tob1.write(b'\x0D\x0A')
                concat([SECONDS,NANOSECONDS,RECORD,df],axis=1).to_records(index=False).tofile(tob1)
        def __init__(self,*args,tz='local',**kwds):
            super().__init__(*args,**kwds)
            self.tz = datetime.datetime.now().astimezone().tzinfo if tz == 'local' else tz
        @property
        def df(self):
            if self.df_ is None:
                if self.ftype == 'TOB1':
                    self.df_ = self.tob1_dataframe()
                elif self.ftype == 'TOA5':
                    self.df_ = self.toa5_dataframe()
                else:
                    self.df_ = self.to_dataframe()
            return self.df_
        def to_dataframe(self,coerce_float=True):
            df = DataFrame.from_records(self.readlines(),columns=self.columns,index=TOB.timestamp_name_,coerce_float=coerce_float)
            df.set_index(DatetimeIndex(df.index,tz=self.tz),inplace=True)
            return df
        def to_numpy(self):
            if self.ftype != 'TOB1':
                return
            def fp2(d):
                return (0x1FFF & d)*(0.1**((0x6000 & d) >> 13))*(-1)**((0x8000 & d) >> 15)
            def dtype_mapper(t,new=False):
                if t.type.startswith('ASCII'):
                    tfmt = f'<{t.size}S'
                elif t.type == 'FP2':
                    tfmt = '<f4' if new else '>i2'
                else:
                    tfmt = TOB.datatype[t.type]['type']
                    if new:
                        tfmt = tfmt.replace('>','<')
                return (t.name,tfmt)
            hfmt = [('SECONDS_','<L'),('NANOSECONDS_','<L')]
            if self.headers[1][2] == 'RECORD':
                hfmt.append((TOB.record_name_,'<L'))
            with TOB.open_file(self.fp) as f:
                nd = fromfile(f,dtype=[*hfmt,*map(dtype_mapper, self.dtype)],offset=self.header_offset)
                for t in self.dtype:
                    if t.type == 'FP2':
                        nd[t.name] = fp2(nd[t.name])
                return nd.astype([*hfmt,*map(lambda x: dtype_mapper(x,new=True), self.dtype)])
        def tob1_dataframe(self,coerce_float=True):
            df = DataFrame.from_records(self.to_numpy(),coerce_float=coerce_float)
            idx = DatetimeIndex(TOB.csepoch+to_timedelta(df.SECONDS_,unit='s')+to_timedelta(df.NANOSECONDS_,unit='ns'),name=TOB.timestamp_name_,tz=self.tz)
            df.set_index(idx,inplace=True)
            df.drop(columns=['SECONDS_','NANOSECONDS_'],inplace=True)
            return df
        def tob1_sqlite3(self,of,table):
            nd = self.to_numpy()
            idx = datetime64(TOB.csepoch) + nd['SECONDS_'].astype('timedelta64[s]') + nd['NANOSECONDS_'].astype('timedelta64[ns]')
            dcol = [t for t in self.columns if t not in (TOB.timestamp_name_,)]
            def to_sql():
                for i,d,dt in nditer([datetime_as_string(idx,'us'),nd[dcol],idx.astype('int64')/1e6 - 9*3600000]):
                    yield tuple((str(i),*d.tolist(),int(dt)))
            self.to_sqlite3(of,table,to_sql,index_col='?{}'.format(len(dcol)+2))
        def toa5_dataframe(self):
            df = read_csv(self.fp,skiprows=4,header=None,names=self.columns,parse_dates=[TOB.timestamp_name_],index_col=TOB.timestamp_name_,na_values=['NAN','nan','NaN','INF','inf','-INF','-inf'])
            df.set_index(df.index.tz_localize(self.tz),inplace=True)
            return df
        def to_tob1(self,of):
            TOBdf.df_tob1(self.df,of,headers=self.headers)
except (ModuleNotFoundError,ImportError):
    pass
def main():
    from importlib.util import find_spec
    if find_spec('pandas'):
        from .tob import TOBdf as TOB
    else:
        from .tob import TOB
    import argparse
    argp = argparse.ArgumentParser()
    argp.add_argument('--outfile',default=':header:')
    argp.add_argument('--append',action='store_true')
    argp.add_argument('--table',default='%t',metavar='store table name; default:tob')
    argp.add_argument('--glob',default='*.dat',metavar='glob pattern; default:*.dat')
    argp.add_argument('--removesuffix',default=[],nargs='*',help='removesuffix from column name')
    argp.add_argument('--rename',default=[],nargs='*',help='rename[replace] column name')
    argp.add_argument('infile', nargs='+')
    args = argp.parse_args()
    if args.infile is None:
        print('no input file')
        argp.print_usage()
        argp.exit()
    infile = list(map(pathlib.Path,args.infile))
    files = [ii for i in infile for ii in (sorted(i.glob(args.glob),key=lambda x: x.name) if i.is_dir() else [i])]
    for f in files:
        print(f.name)
        tob = TOB(f,removesuffix=args.removesuffix,rename=args.rename)
        if args.outfile == ':header:':
            print(tob.headers)
            print(tob.columns)
        else:
            ofname = args.outfile.replace('%n',tob.fp.stem).replace('%t',tob.table)
            if args.outfile.lower().endswith('.csv'):
                tob.to_csv(ofname,args.append)
            elif args.outfile.lower().endswith(('.db','.sqlite3')):
                tbname = args.table.replace('%n',tob.fp.stem).replace('%t',tob.table)
                tob.to_sqlite3(ofname,tbname)
            elif args.outfile.lower().endswith(('.tob','.tob1')):
                TOB.df_tob1(tob.df,ofname,headers=None,units={},methods={})
if __name__ == '__main__':
    main()
