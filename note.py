To convert popoolation-style files from TB into sync files,

def fill_with_ref(x):
    e = ['0','0','0','0','0','0']
    e['ATCG'.index(x.Ref)] = '1'
    e= ":".join(e)
    return x.fillna(e)

x = pd.read_excel('TB_noheader_popoolation.xlsx')
p = [pd.read_excel('TB_noheader_popoolation.xlsx', sheet) for sheet in x.sheet_names]
x = x.sort_values('Position')
x = reduce(lambda d1, d2: merge_the_two(d1, d2), p[:5])
x = x.apply(lambda q: fill_with_ref(q), axis=1)
x.to_csv('TB.sync', sep='\t', index=False)