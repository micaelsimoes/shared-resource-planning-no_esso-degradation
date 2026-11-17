"""Marked copy: manuscript_submitted/main.tex vs the current revised main.tex (expert, 2026-11-12).
Usage: python3 make_marked_copy.py <submitted main.tex> <revised main.tex> <latexdiff-so> <out.tex>
Steps: strip \textcolor{blue}{...} wrappers (balanced braces); remove whitespace-only lines left inside equation /
subequations / align; latexdiff --math-markup=whole with floats as whole blocks; post-process emergencystretch and
references to labels that no longer exist (printed as [label]). Compile with ulem available (Overleaf has it)."""
import re, subprocess, sys
sub, rev, ld, out = sys.argv[1:5]
def strip_blue(s):
    o=[]; i=0; key='\\textcolor{blue}{'
    while True:
        j=s.find(key,i)
        if j<0: o.append(s[i:]); break
        o.append(s[i:j]); k=j+len(key); d=1; p=k
        while d and p<len(s):
            c=s[p]
            if c=='\\': p+=2; continue
            if c=='{': d+=1
            elif c=='}': d-=1
            p+=1
        o.append(s[k:p-1]); i=p
    return ''.join(o)
r=strip_blue(open(rev,encoding='utf-8').read())
r=re.sub(r'\\begin\{(equation|subequations|align)\*?\}.*?\\end\{\1\*?\}',lambda m:'\n'.join(l for l in m.group(0).split('\n') if l.strip()),r,flags=re.S)
open('rev_nocolor.tex','w',encoding='utf-8').write(r)
m=subprocess.run(['perl',ld,'--math-markup=whole','--config',r'PICTUREENV=(?:picture|DIFnomarkup|table|figure|algorithm|longtable)[\w\d*@]*',sub,'rev_nocolor.tex'],capture_output=True,text=True,check=True).stdout
m=m.replace('\\emergencystretch\\DIFadd{=2em\n    }','\\emergencystretch=2em')
defined=set(re.findall(r'\\label\{([^}]*)\}',r)); refs=set(re.findall(r'\\(?:eq)?ref\{([^}]*)\}',m))
for lab in refs-defined:
    m=re.sub(r'\\(?:eq)?ref\{'+re.escape(lab)+r'\}','[\\\\texttt{'+lab.replace('_','\\\\_')+'}]',m)
open(out,'w',encoding='utf-8').write(m); print('written',out,'removed-label refs:',len(refs-defined))
