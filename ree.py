import re


def re_before(string,seq):
    
    return re.findall('(.*?)'+seq, string)[0]

def re_after(string,seq):
    
    return re.findall('(?<='+seq+').*', string)[0]
