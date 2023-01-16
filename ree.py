import re


def re_before(string,seq):
    
    return re.findall('(.*?)'+seq, string)[0]

def re_after(string,seq):
    
    return re.findall('(?<='+seq+').*', string)[0]



def extra_processing(df,txt_col):
    
    df_m = df.copy()
    
    df_m[txt_col] = df_m[txt_col].apply(lambda x: x.lower())
    
    
    df_m[txt_col] = df_m[txt_col].str.replace(r"what's", "what is ")    
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'ve", " have ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"can't", "cannot ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"n't", " not ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"i'm", "i am ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'re", " are ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'d", " would ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'ll", " will ")
    
    df_m[txt_col] = df_m[txt_col].str.replace(r"what's", "what is ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'s", " is ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'ve", " have ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"can't", "cannot ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"n't", " not ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"i'm", "i am ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'re", " are ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'d", " would ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'ll", " will ")
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'scuse", " excuse ")
    
    
    df_m[txt_col] = df_m[txt_col].str.replace(r"pennnis",'dick')
    df_m[txt_col] = df_m[txt_col].str.replace(r"penis",'dick')
    df_m[txt_col] = df_m[txt_col].str.replace(r"pennis",'dick')
    df_m[txt_col] = df_m[txt_col].str.replace(r"pensnsnniensnsn",'dick')
    df_m[txt_col] = df_m[txt_col].str.replace(r"pneis",'dick')
    df_m[txt_col] = df_m[txt_col].str.replace(r"suckersyou",'suckers you')
    df_m[txt_col] = df_m[txt_col].str.replace(r"retardedyour",'retarded your')

        # slang
    df_m[txt_col] = df_m[txt_col].str.replace(r'btw','by the way')


    df_m[txt_col] = df_m[txt_col].str.replace(r"centraliststupid",'centralist stupid')
    df_m[txt_col] = df_m[txt_col].str.replace(r"yourselfgo",'your self go')
    df_m[txt_col] = df_m[txt_col].str.replace(r"youbollocks",'you bollocks')
    df_m[txt_col] = df_m[txt_col].str.replace(r"cuntbag",'cunt bag')
    df_m[txt_col] = df_m[txt_col].str.replace(r"cuntbag",'cunt bag')
    
    
    df_m[txt_col] = df_m[txt_col].str.replace(r'\.',' . ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\?',' ? ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\,',' , ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\!',' ! ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\=',' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\@',' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\#',' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r'\"',' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\'",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\-",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\:",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\;",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\{",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\}",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\d+",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\–",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\/",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\_",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\]",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\[",' ')

    df_m[txt_col] = df_m[txt_col].str.replace(r"\~",' ')
    df_m[txt_col] = df_m[txt_col].str.replace(r"\>",' ')


    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*ing",'fucking')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*cking",'fucking')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*king",'fucking')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*ucking",'fucking')
    df_m[txt_col] = df_m[txt_col].str.replace(r"fuc\*in",'fuckin')


    df_m[txt_col] = df_m[txt_col].str.replace(r"fc\*k",'fuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*",'fuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*k",'fuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"fc\*k",'fuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"fu\*k",'fuck')

    df_m[txt_col] = df_m[txt_col].str.replace(r"fu\*ker",'fucker')
    df_m[txt_col] = df_m[txt_col].str.replace(r"f\*cker",'fucker')

    df_m[txt_col] = df_m[txt_col].str.replace(r"b\*tch",'bitch')
    df_m[txt_col] = df_m[txt_col].str.replace(r"b\*ch",'bitch')

    df_m[txt_col] = df_m[txt_col].str.replace(r"sh\*",'shit')
    df_m[txt_col] = df_m[txt_col].str.replace(r"sh\*t",'shit')
    df_m[txt_col] = df_m[txt_col].str.replace(r"sh\*ting",'shiting')

    df_m[txt_col] = df_m[txt_col].str.replace(r"sh\*ting",'shiting')


    df_m[txt_col] = df_m[txt_col].str.replace(r"fu\*k",'fuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"b\*",'bitch')
    df_m[txt_col] = df_m[txt_col].str.replace(r"b\*h",'bitch')
    df_m[txt_col] = df_m[txt_col].str.replace(r"bulls\*t",'bullshit')
    df_m[txt_col] = df_m[txt_col].str.replace(r"as\*ole",'asshole')
    df_m[txt_col] = df_m[txt_col].str.replace(r"a\*hole",'asshole')
    df_m[txt_col] = df_m[txt_col].str.replace(r"starfu\*k",'starfuck')
    df_m[txt_col] = df_m[txt_col].str.replace(r"pri\*k",'prick')

    df_m[txt_col] = df_m[txt_col].str.replace(r"wh\*r\*",'whore')


    df_m[txt_col] = df_m[txt_col].str.replace(r'([a-zA-Z])\1\1{2,}',r'\1') 

    df[txt_col] = df[txt_col].str.replace(r'\n',' ')
    
    df_m[txt_col] = df_m[txt_col].str.replace(r' +',r' ') 


    df_m[txt_col] = df_m[txt_col].apply(lambda x: x.strip())
    
    
    return df_m



def remove_new_line(x):
    
    x = x.strip()
    
    x = re.sub('\n','',x)
    
    return x.strip()

def remove_new_qoats_start_finish(x):
    
    if x[0] in ["'",'"']:
        
        x = x[1:]
        
    if x[-1] in ["'",'"']:
        
        x = x[:-1]
        
    return x


def check_names(x):
    
    checks = re.findall(r'[0-9.]+',x)
    
    if len(checks) == 0:
        
        return x


    for i in checks:
        
        if len(re.findall(r'[0-9]+',i)) > 0 and '.' in i:
            
            temp = Counter(i)

            if temp['.'] >2 :
                
                return re.sub(i,'',x)
        
    return x




def clean_repeated_punc(x):
    
    puncs = "!#$%&\'()*+,-.:;<=>?@[]_`{|}~"

    reppeating = re.findall(r'(.)\1+',x)

    for i in reppeating :

        if i in puncs:

            x =  re.sub(r'(['+i+'])\\1+', r'\1', x)
            
    return x



def clean_ulrs(x):
    
    cases = '()?&+'
    
    to_clean = re.findall(r'\(http:[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+',x) + re.findall(r'\(http:[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+ ',x) \
     + re.findall(r'http:[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+',x) + re.findall(r'http:[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+ ',x) + \
    re.findall(r'http[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+',x) + re.findall(r'http[a-zA-Z0-9/!#$%&\'()*+,-.;<=>?@~_]+ ',x)
    
    for ur in to_clean:
        try:
            
            for j in cases:
                if j in ur:
                    
                    ur = re.sub('\\'+j,'\\'+j,ur)
            
            x = re.sub(ur,' ',x)

        except:
            
            continue
        
        
    return x.strip()
    
    
    
def extra_clean(x):
    
    extra_characters = ['\\\\','\|','\)','\(']
    
    
    for h in extra_characters:

        x = re.sub(h,' ',x)
        
    return x.strip()
